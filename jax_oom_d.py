"""A program to reproduce the OOM condition with Jax training.

On the TPU VM:
mkdir -p ./jax_oom_d && python3 jax_oom_d.py --dir=./jax_oom_d 2>&1 | tee ./jax_oom_d/log
"""
import os.path
import time
from typing import Any, Callable, Dict, NamedTuple, Optional, Union

import jax
import numpy as np
from absl import app, flags
from absl import logging
from jax import numpy as jnp
from jax.experimental import PartitionSpec
from jax.experimental import maps
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit

import checkpointer
import config as config_lib
import learner
import param_init
import summary_writer
from module import BaseLayer, NestedParameterSpec, ParameterSpec
from module import functional as F, Module, NestedTensor, NestedPartitionSpec
from utils import Tensor, tree_paths

flags.DEFINE_string(
    "dir",
    None,
    "The root directory of the trainer. "
    "Checkpoints will be stored in <dir>/checkpoints. "
    "Summaries will be stored in <dir>/summaries.",
    required=True,
)
flags.DEFINE_integer("jax_profiler_port", None, "If not None, the profiler port.")

FLAGS = flags.FLAGS


class DummyInput(Module):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("global_batch_size", 256, "The batch size.")
        cfg.define(
            "total_num_batches",
            None,
            "The total number of batches. If None, unlimited.",
        )
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent=None):
        super().__init__(cfg, parent=parent)
        self._prng_key = jax.random.PRNGKey(1)
        self._num_batches = 0

    def __iter__(self):
        self._num_batches = 0
        return self

    def __next__(self):
        cfg = self.config
        self._num_batches += 1
        if (
                cfg.total_num_batches is not None
                and self._num_batches > cfg.total_num_batches
        ):
            raise StopIteration()
        self._prng_key, image_key, label_key = jax.random.split(self._prng_key, 3)
        return dict(
            image=jax.random.randint(
                image_key,
                shape=[cfg.global_batch_size, 224, 224, 3],
                minval=0,
                maxval=256,
                dtype=np.int32,
            ),
            label=jax.random.randint(
                label_key, shape=[cfg.global_batch_size], minval=0, maxval=1000, dtype=np.int32
            ),
        )


class DummyModel(BaseLayer):

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'scale': ParameterSpec(shape=[], partition_spec=[]),
            'bias': ParameterSpec(shape=[], partition_spec=[]),
        }

    def initialize_parameters_recursively(
            self,
            prng_key: jax.random.KeyArray,
            param_specs: Optional[NestedParameterSpec] = None,
    ) -> NestedTensor:
        return {
            'scale': jnp.ones(shape=[], dtype=jnp.float32),
            'bias': jnp.zeros(shape=[], dtype=jnp.float32),
        }

    def forward(self, image: Tensor, label: Tensor):
        x = image.mean(axis=(1, 2, 3))
        x = x * self.state['scale'] + self.state['bias']
        loss = jnp.abs(x - label.astype(x.dtype)).mean()
        return loss, {}


def no_op_optimizer() -> learner.PartitionedGradientTransformation:
    def no_op_update(updates, state, params):
        logging.info("no_op_update: g=%s s=%s p=%s", updates, state, params)
        updates = jax.tree_map(lambda x: jnp.zeros_like(x), params)
        logging.info("no_op_update: u=%s", updates)
        return updates, state

    return learner.PartitionedGradientTransformation(
        init=lambda x: {},
        update=no_op_update,
        partition=lambda x: {},
    )


def _apply_updates(base, updates):
    if isinstance(updates, jnp.ndarray):
        assert isinstance(base, jnp.ndarray), base
        return updates
    for k, v in updates.items():
        if k not in base:
            base[k] = v
        else:
            base[k] = _apply_updates(base[k], v)
    return base


class _TrainerState(NamedTuple):
    step: Union[Tensor, NestedPartitionSpec]
    model: Union[NestedTensor, NestedPartitionSpec]
    learner: Union[learner.LearnerState, NestedPartitionSpec]


class SpmdTrainer(Module):
    """A base class for running computation (training, evaluation, predictions) on SPMD."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("input", None, "The model input config.")
        cfg.define(
            "summary_writer",
            summary_writer.SummaryWriter.default_config(),
            "The summary writer.",
        )
        cfg.define("model", None, "The model config.")
        cfg.define("learner", None, "The learner config.")
        cfg.define(
            "checkpointer",
            checkpointer.Checkpointer.default_config(),
            "The checkpointer.",
        )
        cfg.define(
            "evalers",
            tuple(),
            "A list/tuple of evaler configs, each must have non-empty and unique names.",
        )
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("input", cfg.input)
        self._add_child("summary_writer", cfg.summary_writer)
        self._state = None
        self._jit_compute = None
        self._add_child("model", cfg.model)
        self._add_child("learner", cfg.learner)
        self._add_child("checkpointer", cfg.checkpointer)
        for evaler_cfg in cfg.evalers:
            self._add_child(evaler_cfg.name, evaler_cfg)

        self._model_param_specs = self.model.create_parameter_specs_recursively()
        self.vlog(3, "Model param specs: %s", self._model_param_specs)
        model_param_partition_specs = jax.tree_map(
            lambda spec: PartitionSpec(*spec.partition_spec), self._model_param_specs
        )
        # for path, spec in flatten_items(model_param_partition_specs):
        #     self.vlog(3, "Model param partition: %s=%s", path, spec)
        self._step_log("Model param partition: %s", model_param_partition_specs)
        learner_state_partition_specs = self.learner.create_state_partition_specs(
            model_param_partition_specs
        )
        self._trainer_state_partition_specs = _TrainerState(
            step=None,
            model=model_param_partition_specs,
            learner=learner_state_partition_specs,
        )
        self._jit_train_step = self._jit(
            self._train_step,
            in_axis_resources=(
                None,  # prng_key
                self._trainer_state_partition_specs,
                self._input_sharding(),
            ),
            out_axis_resources=None,
            donate_argnums=(0, 1, 2),
        )
        for evaler_cfg in cfg.evalers:
            self._children[evaler_cfg.name].init(self.model, model_param_partition_specs)

    def _jit(self, fn: Callable, *, in_axis_resources, out_axis_resources, **kwargs):
        self.vlog(3, "Compiling computation %s", fn)
        if all(device.platform in ("tpu", "gpu") for device in jax.devices()):
            fn = pjit(fn, in_axis_resources=in_axis_resources, out_axis_resources=out_axis_resources, **kwargs)
        else:
            logging.log_first_n(
                logging.INFO,
                "Falling back to jit on %s",
                1,
                [device.platform for device in jax.devices()],
            )
            fn = jax.jit(fn, **kwargs)
        self.vlog(3, "Compiling computation done")
        return fn

    def _input_sharding(self):
        # Shard data along the batch dim.
        return PartitionSpec("data")

    @property
    def step(self):
        if self._state is None:
            return 0
        return self._state.step

    def _step_log(self, msg, *args, **kwargs):
        logging.info(
            "%s process % 3d step % 8d] " + msg,
            self.path(),
            jax.process_index(),
            self.step,
            *args,
            **kwargs)

    def run(self, prng_key: jax.random.KeyArray, max_step: int):
        cfg = self.config
        jax.config.update('jax_log_compiles', True)
        self._step_log("Starting run up to step %s", max_step)
        prng_key, init_key = jax.random.split(prng_key)
        self._init(init_key)

        with jax.profiler.trace(cfg.summary_writer.dir):
            start_time = time.perf_counter()
            num_steps = 0
            for input_batch in self.input:
                prng_key, step_key = jax.random.split(prng_key)
                self.vlog(3, "Start step %s", self.step + 1)
                self._run_step(step_key, input_batch)
                self.vlog(3, "Done step %s", self.step)
                num_steps += 1
                if num_steps % 100 == 0:
                    now = time.perf_counter()
                    self._step_log("Average step time: %s seconds", (now - start_time) / num_steps)
                    num_steps = 0
                    start_time = now
                if self.step >= max_step:
                    self._step_log("Reached max_step=%s. Stopping", max_step)
                    return
            self._step_log("Reached end of inputs. Stopping")

    def _init(self, prng_key: jax.random.KeyArray):
        def _init_state(prng_key: jax.random.KeyArray):
            model_params = self.model.initialize_parameters_recursively(
                prng_key, self._model_param_specs
            )
            learner_params = self.learner.init(model_params)
            return _TrainerState(
                step=jnp.zeros([], dtype=jnp.int64),
                model=model_params,
                learner=learner_params,
            )

        init_computation = self._jit(
            _init_state,
            in_axis_resources=(None,),
            out_axis_resources=self._trainer_state_partition_specs,
        )
        self._step_log("Initializing states")
        self._state = init_computation(prng_key)
        flat_paths, _ = jax.tree_flatten(tree_paths(self._state))
        flat_values, _ = jax.tree_flatten(self._state)
        for path, value in zip(flat_paths, flat_values):
            self._step_log("State: %s=%s(%s)", path, value.dtype, value.shape)

        # Try to restore the latest checkpoint.
        self._state = self.checkpointer.restore(step=None, state=self._state)

    def _run_step(self, prng_key: jax.random.KeyArray, input_batch: Any):
        cfg = self.config
        self.vlog(3, "  train_step: %s", self.step + 1)
        with jax.profiler.StepTraceAnnotation("train", step_num=self.step):
            prng_key, train_key = jax.random.split(prng_key)
            # Note(Jan 2022): pjit currently requires all parameters to be specified as positional args.
            outputs = self._jit_train_step(train_key, self._state, input_batch)
            self._state = outputs["state"]
        if self._state.step % 1 == 0:
            self._step_log(
                "loss=%s aux=%s",
                outputs["loss"],
                jax.tree_map(
                    lambda x: x.item() if x.ndim == 0 else f"T{x.shape}", outputs["aux"]
                ),
            )
        self.vlog(3, "  summary_writer: %s", self.step)
        self.summary_writer(
            self.step, {"loss": outputs["loss"], **outputs["summaries"]}
        )
        self.vlog(3, "  eval: %s", self.step)
        for evaler_cfg in cfg.evalers:
            prng_key, eval_key = jax.random.split(prng_key)
            self._children[evaler_cfg.name].run_step(
                self.step,
                prng_key=eval_key,
                model_params=self._state.model,
            )
        self.vlog(3, "  checkpointer: %s", self.step)
        self.checkpointer.save(step=self.step, state=self._state)

    def _train_step(
            self,
            prng_key: jax.random.KeyArray,
            state: _TrainerState,
            input_batch: Dict[str, Any],
    ):
        prng_key, forward_key, learner_key = jax.random.split(prng_key, 3)

        def _forward(model_parameters, forward_input_batch):
            (loss, aux), model_output_collection = F(
                self.model,
                state=model_parameters,
                is_training=True,
                prng_key=forward_key,
                inputs=forward_input_batch,
            )
            return loss, (aux, model_output_collection)

        _forward_and_grad = jax.value_and_grad(_forward, has_aux=True)
        (loss, (forward_aux, forward_output_collection)), grads = _forward_and_grad(
            state.model, jax.tree_map(lambda x: jnp.asarray(x), input_batch)
        )

        updated_model_params, learner_output_collection = F(
            self.learner,
            method="update",
            state=state.learner,
            is_training=True,
            prng_key=learner_key,
            inputs=dict(step=state.step, model_params=state.model, gradients=grads),
        )
        updated_state = _TrainerState(
            step=state.step + 1,
            model=_apply_updates(
                updated_model_params, forward_output_collection.state_updates
            ),
            learner=learner.LearnerState(**learner_output_collection.state_updates),
        )
        # TODO(ruoming): only retrieve summaries when necessary.
        summaries = dict(
            model=forward_output_collection.summaries,
            learner=learner_output_collection.summaries,
        )
        return dict(
            state=updated_state,
            summaries=summaries,
            loss=loss,
            aux=forward_aux,
        )


def imagenet_trainer_config():
    num_train_examples = 1_281_167
    train_batch_size = 256
    eval_batch_size = 80  # divides 50_000 and can be divided by number of devices (8)
    steps_per_epoch = num_train_examples // train_batch_size

    cfg = SpmdTrainer.default_config()
    cfg.name = "imagenet_trainer"

    # Model and optimization.
    cfg.model = DummyModel.default_config().set(dtype=jnp.float32,
                                                param_init=param_init.DefaultInitializer.default_config())
    cfg.learner = learner.Learner.default_config().set(
        optimizer=config_lib.config_for_function(no_op_optimizer))

    # Training inputs.
    cfg.input = DummyInput.default_config().set(
        global_batch_size=train_batch_size,
    )
    cfg.evalers = []

    # Summaries and checkpoints.
    cfg.checkpointer.dir = os.path.join(FLAGS.dir, "checkpoints")
    cfg.checkpointer.write_every_n_steps = 100000000
    cfg.checkpointer.keep_every_n_steps = cfg.checkpointer.write_every_n_steps * 10
    summary_dir = os.path.join(FLAGS.dir, "summaries")
    cfg.summary_writer.write_every_n_steps = 100000000
    cfg.summary_writer.dir = os.path.join(summary_dir, "train_train")
    return cfg


def run_trainer(trainer_config):
    trainer: SpmdTrainer = trainer_config.instantiate(parent=None)
    prng_key = jax.random.PRNGKey(1)
    mesh_shape = (jax.device_count(), 1)
    devices = mesh_utils.create_device_mesh(mesh_shape)
    mesh = maps.Mesh(devices, ("data", "model"))
    with maps.mesh(mesh.devices, mesh.axis_names):
        trainer.run(prng_key, max_step=1000000000)


def main(argv):
    # Start jax.profiler for Tensorboard and profiling in open source.
    if FLAGS.jax_profiler_port is not None:
        server = jax.profiler.start_server(FLAGS.jax_profiler_port)

    logging.info("Creating trainer config")
    trainer_config = imagenet_trainer_config()
    logging.info("Trainer config: %s", trainer_config.debug_string())
    run_trainer(trainer_config)


if __name__ == "__main__":
    app.run(main)

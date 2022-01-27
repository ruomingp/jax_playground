"""A program to reproduce the OOM condition with Jax training.

On the TPU VM:
mkdir -p ./jax_oom_m && python3 jax_oom_m.py --dir=./jax_oom_m 2>&1 | tee ./jax_oom_m/log

Same as jax_oom_l except without a BaseLayer-based DummyModel.
"""
import os.path
import time
from typing import Any, Callable, Dict, NamedTuple, Optional, Union

import jax
import numpy as np
import optax
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
        # self._prng_key, image_key, label_key = jax.random.split(self._prng_key, 3)
        image_key, label_key = self._prng_key, self._prng_key
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


class DummyModel:

    def parameter_partition_specs(self) -> NestedPartitionSpec:
        return {
            'scale': PartitionSpec(),
            'bias': PartitionSpec(),
        }

    def initialize_parameters_recursively(self, prng_key) -> NestedTensor:
        return {
            'scale': jnp.ones(shape=[], dtype=jnp.float32),
            'bias': jnp.zeros(shape=[], dtype=jnp.float32),
        }

    def forward(self, state: NestedTensor, image: Tensor, label: Tensor):
        x = image.mean(axis=(1, 2, 3))
        x = x * state['scale'] + state['bias']
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
        cfg.define("model", None, "The model config.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("input", cfg.input)
        self._state = None
        self._jit_compute = None
        self.model = DummyModel()
        self.learner = no_op_optimizer()

        model_param_partition_specs = self.model.parameter_partition_specs()
        self._step_log("Model param partition: %s", model_param_partition_specs)
        learner_state_partition_specs = self.learner.partition(
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

    def _jit(self, fn: Callable, *, in_axis_resources, out_axis_resources, **kwargs):
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
        if self.step % 1 != 0:
            return
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
        # prng_key, init_key = jax.random.split(prng_key)
        init_key = prng_key
        self._init(init_key)

        with jax.profiler.trace(FLAGS.dir):
            start_time = time.perf_counter()
            num_steps = 0
            for input_batch in self.input:
                # prng_key, step_key = jax.random.split(prng_key)
                step_key = prng_key
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
            model_params = self.model.initialize_parameters_recursively(prng_key)
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

    def _run_step(self, prng_key: jax.random.KeyArray, input_batch: Any):
        cfg = self.config
        self.vlog(3, "  train_step: %s", self.step + 1)
        with jax.profiler.StepTraceAnnotation("train", step_num=self.step):
            # prng_key, train_key = jax.random.split(prng_key)
            train_key = prng_key
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

    def _train_step(
            self,
            prng_key: jax.random.KeyArray,
            state: _TrainerState,
            input_batch: Dict[str, Any],
    ):
        # prng_key, forward_key, learner_key = jax.random.split(prng_key, 3)
        prng_key, forward_key, learner_key = prng_key, prng_key, prng_key

        def _forward(model_parameters, forward_input_batch):
            loss, aux = self.model.forward(model_parameters, **forward_input_batch)
            return loss, aux

        _forward_and_grad = jax.value_and_grad(_forward, has_aux=True)
        (loss, forward_aux), grads = _forward_and_grad(
            state.model, jax.tree_map(lambda x: jnp.asarray(x), input_batch)
        )

        updates, updated_learner_state = self.learner.update(grads, state=state.learner, params=state.model)
        updated_model_params = optax.apply_updates(state.model, updates)
        updated_state = _TrainerState(
            step=state.step + 1,
            # model=_apply_updates(updated_model_params, forward_output_collection.state_updates),
            model=updated_model_params,
            learner=updated_learner_state,
        )
        return dict(
            state=updated_state,
            loss=loss,
            aux=forward_aux,
        )


def imagenet_trainer_config():
    num_train_examples = 1_281_167
    train_batch_size = 256

    cfg = SpmdTrainer.default_config()
    cfg.name = "imagenet_trainer"

    # Model and optimization.
    cfg.model = DummyModel.default_config().set(dtype=jnp.float32,
                                                param_init=param_init.DefaultInitializer.default_config())
    # Training inputs.
    cfg.input = DummyInput.default_config().set(
        global_batch_size=train_batch_size,
    )
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

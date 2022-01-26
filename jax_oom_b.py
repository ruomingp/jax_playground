"""A launcher to train ResNet-18 on ImageNet.

On the TPU VM:
gs_bucket=permanent-us-central1-q5loch
exp=$(date +%F-%H-%M)
dir=gs://${gs_bucket}/${USER}/experiments/jax_oom_a
echo $dir
python3 jax_oom_a.py --dir=$dir 2>&1 | tee log.${exp}
"""
import os.path
from typing import Any, Callable, Dict, NamedTuple, Optional, Union

import jax  # jax must be imported before tensorflow!
import numpy as np
import optax
from absl import app, flags, logging
from jax import numpy as jnp
from jax.experimental import PartitionSpec
from jax.experimental import maps
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit

import config as config_lib
import learner
import param_init
from module import BaseLayer, Module, NestedParameterSpec, ParameterSpec
from trainer import SpmdTrainer

Tensor = jnp.ndarray
# Recursive type annotations not supported by pytype yet.
NestedTree = Union[Any, Dict[str, Any]]  # Union[Any, Dict[str, "NestedTree"]]
NestedTensor = Union[Tensor, Dict[str, Any]]  # Union[Tensor, Dict[str, "NestedTensor"]]
NestedPartitionSpec = Optional[Union[PartitionSpec, Dict[str, Any]]]


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

    def parameter_partition_specs(self) -> NestedPartitionSpec:
        return {
            'scale': [],
            'bias': [],
        }

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

    def forward(self, state: NestedTensor, image: Tensor, label: Tensor):
        x = image.mean(axis=(1, 2, 3))
        x = x * state['scale'] + state['bias']
        loss = jnp.abs(x - label.astype(x.dtype)).mean()
        return loss, {}

    def forward0(self, image: Tensor, label: Tensor):
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


class _TrainerState(NamedTuple):
    step: Union[Tensor, NestedPartitionSpec]
    model: Union[NestedTensor, NestedPartitionSpec]
    learner: Union[NestedTensor, NestedPartitionSpec]


class Trainer:

    def __init__(self, input_config, model_config):
        self.input = input_config.set(name="input").instantiate(parent=None)
        self.model = model_config.set(name="model").instantiate(parent=None)
        self.optimizer = no_op_optimizer()
        self._state = None

        model_param_partition_specs = self.model.parameter_partition_specs()
        self._step_log("Model param partition: %s", model_param_partition_specs)
        self._trainer_state_partition_specs = _TrainerState(
            step=None,
            model=model_param_partition_specs,
            learner=None,
        )
        self._jit_train_step = self._jit(
            self._train_step,
            in_axis_resources=(
                self._trainer_state_partition_specs,
                PartitionSpec("data"),
            ),
            out_axis_resources=None,
            donate_argnums=(0, 1),
        )

    @property
    def step(self):
        if self._state is None:
            return 0
        return self._state.step

    def _step_log(self, msg, *args, **kwargs):
        logging.info(
            "process % 3d step % 8d] " + msg,
            jax.process_index(),
            self.step,
            *args,
            **kwargs)

    def run(self, prng_key: jax.random.KeyArray, max_step: int):
        jax.config.update('jax_log_compiles', True)
        self._step_log("Starting run up to step %s", max_step)
        self._init(prng_key)

        with jax.profiler.trace(FLAGS.dir):
            for input_batch in self.input:
                self._run_step(input_batch)

    def _init(self, prng_key):
        def _init_state(prng_key):
            model_params = self.model.initialize_parameters_recursively(prng_key)
            learner_params = self.optimizer.init(model_params)
            return _TrainerState(
                step=jnp.zeros([], dtype=jnp.int64),
                model=model_params,
                learner=learner_params,
            )

        init_computation = self._jit(
            _init_state,
            in_axis_resources=[None],
            out_axis_resources=self._trainer_state_partition_specs,
        )
        self._step_log("Initializing states")
        self._state = init_computation(prng_key)

    def _run_step(self, input_batch: Any):
        with jax.profiler.StepTraceAnnotation("train", step_num=self.step):
            # Note(Jan 2022): pjit currently requires all parameters to be specified as positional args.
            output, self._state = self._jit_train_step(self._state, input_batch)
        self._step_log("output=%s", output)

    def _train_step(
            self,
            state: _TrainerState,
            input_batch: Dict[str, Any],
    ):
        def _forward(model_parameters, forward_input_batch):
            return self.model.forward(
                state=model_parameters,
                **forward_input_batch,
            )

        _forward_and_grad = jax.value_and_grad(_forward, has_aux=True)
        (loss, aux), grads = _forward_and_grad(
            state.model, jax.tree_map(lambda x: jnp.asarray(x), input_batch)
        )

        updates, updated_learner_state = self.optimizer.update(grads, state=state.learner, params=state.model)
        updated_model = optax.apply_updates(state.model, updates)
        updated_state = _TrainerState(
            step=state.step + 1,
            model=updated_model,
            learner=updated_learner_state
        )
        return (loss, aux), updated_state

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
        return fn


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
    trainer = Trainer(input_config=trainer_config.input,
                      model_config=trainer_config.model)
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

    trainer_config = imagenet_trainer_config()
    logging.info("Trainer config: %s", trainer_config.debug_string())
    run_trainer(trainer_config)


if __name__ == "__main__":
    app.run(main)

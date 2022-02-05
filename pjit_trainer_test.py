"""A program to reproduce the OOM condition with Jax training.

On the TPU VM:
python3 pjit_trainer_test.py --dir=gs://permanent-us-central1-q5loch/r_pang_apple_com/experiments/pjit_trainer_test.a
"""

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

Tensor = jnp.ndarray
# Recursive type annotations not supported by pytype yet.
NestedTree = Union[Any, Dict[str, Any]]  # Union[Any, Dict[str, "NestedTree"]]
NestedTensor = Union[Tensor, Dict[str, Any]]  # Union[Tensor, Dict[str, "NestedTensor"]]
NestedPartitionSpec = Optional[Union[PartitionSpec, Dict[str, Any]]]


flags.DEFINE_string(
    "dir",
    None,
    "The directory of the trainer profiles.",
    required=True,
)
flags.DEFINE_list(
    "start_trace_steps",
    [10, 20, 30],
    "Steps on which we start profiler tracing."
)

FLAGS = flags.FLAGS


TransformPartitionSpecFn = Callable[[NestedPartitionSpec], NestedPartitionSpec]

class PartitionedGradientTransformation(NamedTuple):
    init: optax.TransformInitFn
    update: optax.TransformUpdateFn
    partition: TransformPartitionSpecFn


def no_op_optimizer() -> PartitionedGradientTransformation:
    def no_op_update(updates, state, params):
        logging.info("no_op_update: g=%s s=%s p=%s", updates, state, params)
        updates = jax.tree_map(lambda x: jnp.zeros_like(x), params)
        logging.info("no_op_update: u=%s", updates)
        return updates, state

    return PartitionedGradientTransformation(
        init=lambda x: {},
        update=no_op_update,
        partition=lambda x: {},
    )


class DummyInput:

    def __init__(self):
        self._prng_key = jax.random.PRNGKey(1)
        self._global_batch_size = 256
        self._num_batches = 0

    def __iter__(self):
        self._num_batches = 0
        return self

    def __next__(self):
        self._num_batches += 1
        self._prng_key, image_key, label_key = jax.random.split(self._prng_key, 3)
        return dict(
            image=jax.random.randint(
                image_key,
                shape=[self._global_batch_size, 224, 224, 3],
                minval=0,
                maxval=256,
                dtype=np.int32,
            ),
            label=jax.random.randint(
                label_key, shape=[self._global_batch_size], minval=0, maxval=1000, dtype=np.int32
            ),
        )


class DummyModel:

    def parameter_partition_specs(self) -> NestedPartitionSpec:
        return {
            'scale': PartitionSpec(),
            'bias': PartitionSpec(),
        }

    def initialize_parameters_recursively(self) -> NestedTensor:
        return {
            'scale': jnp.ones(shape=[], dtype=jnp.float32),
            'bias': jnp.zeros(shape=[], dtype=jnp.float32),
        }

    def forward(self, state: NestedTensor, image: Tensor, label: Tensor):
        x = image.mean(axis=(1, 2, 3))
        x = x * state['scale'] + state['bias']
        loss = jnp.abs(x - label.astype(x.dtype)).mean()
        return loss, {}


class _TrainerState(NamedTuple):
    step: Union[Tensor, NestedPartitionSpec]
    model: Union[NestedTensor, NestedPartitionSpec]
    learner: Union[NestedTensor, NestedPartitionSpec]


class Trainer:

    def __init__(self):
        self.input = DummyInput()
        self.model = DummyModel()
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

    def run(self):
        jax.config.update('jax_log_compiles', True)
        self._init()

        num_steps = 0
        for input_batch in self.input:
            if num_steps in FLAGS.start_trace_steps:
                logging.info("Start tracing...")
                jax.profiler.start_trace(FLAGS.dir)
            if num_steps - 3 in FLAGS.start_trace_steps:
                jax.profiler.stop_trace()
                logging.info("Stopped tracing...")
            self._run_step(input_batch)
            num_steps += 1

    def _init(self):
        def _init_state():
            model_params = self.model.initialize_parameters_recursively()
            learner_params = self.optimizer.init(model_params)
            return _TrainerState(
                step=jnp.zeros([], dtype=jnp.int64),
                model=model_params,
                learner=learner_params,
            )

        init_computation = self._jit(
            _init_state,
            in_axis_resources=[],
            out_axis_resources=self._trainer_state_partition_specs,
        )
        self._step_log("Initializing states")
        self._state = init_computation()

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


def main(argv):
    trainer = Trainer()
    mesh_shape = (jax.device_count(), 1)
    devices = mesh_utils.create_device_mesh(mesh_shape)
    mesh = maps.Mesh(devices, ("data", "model"))
    with maps.mesh(mesh.devices, mesh.axis_names):
        trainer.run()


if __name__ == "__main__":
    app.run(main)

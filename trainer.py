from typing import Any, Dict, NamedTuple, Optional

import jax
from absl import logging
from jax import numpy as jnp
from jax.experimental import PartitionSpec
from jax.experimental.pjit import pjit

import checkpointer
import config as config_lib
import learner
import module
import summary_writer
from module import InvocationContext, Module, NestedParameters, tree_paths


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


class AbstractRunner(Module):

    def init(self, prng_key: jax.random.KeyArray):
        raise NotImplementedError(type(self))

    def run_step(self, step: int, prng_key: jax.random.KeyArray):
        raise NotImplementedError(type(self))


class TrainerEvaler(AbstractRunner):
    """A composite runner that runs trainer and evalers."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("trainer", None, "The trainer config.")
        cfg.define("evalers", None, "One or a list of evaler configs.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config


class _SpmdRunner(AbstractRunner):
    """A base class for running computation (training, evaluation, predictions) on SPMD."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("input", None, "The model input config.")
        cfg.define("model", None, "The model config.")
        cfg.define("summary_writer", summary_writer.SummaryWriter.default_config(), "The summary writer.")
        cfg.define("checkpointer", checkpointer.Checkpointer.default_config(), "The checkpointer.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        # self._add_child('input', cfg.input)
        self.input = cfg.input.instantiate()
        self._add_child('model', cfg.model)
        self._add_child('summary_writer', cfg.summary_writer)
        self._add_child('checkpointer', cfg.checkpointer)
        self._state = None
        self._jit_compute = None

    def init(self, prng_key: jax.random.KeyArray):
        logging.info("Compiling init_state")
        init_computation = self._jit(self._init_state, in_axis_resources=(None,), out_axis_resources=None)
        logging.info("Initializing states")
        self._state = init_computation(prng_key)
        flat_paths, _ = jax.tree_flatten(tree_paths(self._state))
        flat_values, _ = jax.tree_flatten(self._state)
        for path, value in zip(flat_paths, flat_values):
            logging.info("%s state: %s=%s(%s)", self.path(), path, value.dtype, value.shape)
        logging.info("Compiling computation")
        self._jit_compute = self._jit(self._compute,
                                      in_axis_resources=(
                                          None,  # prng_key
                                          self._state_sharding(),
                                          self._input_sharding(),
                                      ),
                                      out_axis_resources=self._compute_output_sharding())
        logging.info("Compiling computation done")

    def _init_state(self, prng_key: jax.random.KeyArray):
        raise NotImplementedError(type(self))

    def run_step(self, step: int, prng_key: jax.random.KeyArray):
        raise NotImplementedError(type(self))

    def _jit(self, fn, **kwargs):
        if all(device.platform in ('tpu', 'gpu') for device in jax.devices()):
            return pjit(fn, **kwargs)
        else:
            logging.info('Falling back to jit on %s', [device.platform for device in jax.devices()])
            return jax.jit(fn)

    def _compute(self, prng_key: jax.random.KeyArray, state: Any, input_batch: Any) -> Any:
        """The computation to be jit."""
        raise NotImplementedError(type(self))

    def _state_sharding(self):
        # TODO(ruoming): support state sharding.
        return None

    def _input_sharding(self):
        # Shard data along the batch dim.
        return PartitionSpec('data')

    def _compute_output_sharding(self):
        return None


class _TrainerState(NamedTuple):
    model: NestedParameters
    learner: learner.LearnerState


class SpmdTrainer(_SpmdRunner):

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("learner", None, "The learner config.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child('learner', cfg.learner)

    def _init_state(self, prng_key: jax.random.KeyArray):
            model_params = self.model.initialize_parameters_recursively(prng_key)
            learner_params = self.learner.init(model_params)
            return _TrainerState(model=model_params, learner=learner_params)

    def run_step(self, step: int, prng_key: jax.random.KeyArray):
        self.checkpointer.save(step=step, state=self._state)
        input_batch = next(self.input)
        # Note(Jan 2022): pjit currently requires all parameters to be specified as positional args.
        outputs = self._jit_compute(prng_key, self._state, input_batch)
        logging.info("Process % 3d step % 8d: loss=%s aux=%s",
                     jax.process_index(), step, outputs["loss"], outputs["aux"])
        self._state = outputs["state"]
        self.summary_writer(step, outputs["summaries"])

    def _compute(self, prng_key: jax.random.KeyArray, state: _TrainerState, input_batch: Dict[str, Any]):
        forward_key, learner_key = jax.random.split(prng_key)
        del prng_key

        def _forward(model_parameters, input_batch):
            forward_context: InvocationContext = self.model.make_invocation_context(
                parameters=model_parameters, is_training=True, prng_key=forward_key)
            with module.root_context(forward_context):
                loss, aux = self.model(**input_batch)
            return loss, dict(aux=aux, parameter_updates=forward_context.get_parameter_updates(),
                              summaries=forward_context.get_summaries())

        _forward_and_grad = jax.value_and_grad(_forward, has_aux=True)
        (loss, forward_outputs), grads = _forward_and_grad(state.model,
                                                           jax.tree_map(lambda x: jnp.asarray(x), input_batch))

        learner_context: InvocationContext = self.learner.make_invocation_context(
            parameters=None, is_training=True, prng_key=learner_key)
        with module.root_context(learner_context):
            updated_learner_state, updated_model_params = self.learner.update(
                state.learner, model_params=state.model, gradients=grads)
        updated_state = _TrainerState(
            model=_apply_updates(updated_model_params, forward_outputs["parameter_updates"]),
            learner=updated_learner_state)
        # TODO(ruoming): only retrieve summaries when necessary.
        summaries = dict(model=forward_outputs["summaries"], learner=learner_context.get_summaries())
        return dict(state=updated_state, summaries=summaries, loss=loss, aux=forward_outputs["aux"])


class _EvalerState(NamedTuple):
    model: NestedParameters


class SpmdEvaler(_SpmdRunner):

    def _init_state(self, prng_key: jax.random.KeyArray):
        model_params = self.model.initialize_parameters_recursively(prng_key)
        return _EvalerState(model=model_params)

    def run_step(self, step: int, prng_key: jax.random.KeyArray):
        self._state = self.checkpointer.restore(step=step, state=self._state)
        prng_key, init_key = jax.random.split(prng_key)
        metrics = self._init_metrics()
        num_batches = 0
        for input_batch in self.input:
            prng_key, batch_key = jax.random.split(prng_key)
            outputs = self._jit_compute(batch_key, self._state, input_batch)
            logging.info("Process % 3d step % 8d: loss=%s aux=%s",
                         jax.process_index(), step, outputs["loss"], outputs["aux"])
            if num_batches == 0:
                self.summary_writer(step, outputs["summaries"])
            num_batches += 1
            metrics.update(input_batch, outputs)
        self.summary_writer(step, metrics.summaries())

    def _compute(self, prng_key: jax.random.KeyArray, state: _EvalerState, input_batch: Dict[str, Any]):
        forward_context: InvocationContext = self.model.make_invocation_context(
            parameters=state.model, is_training=False, prng_key=prng_key)
        with module.root_context(forward_context):
            loss, aux = self.model(**input_batch)
        return loss, dict(aux=aux, summaries=forward_context.get_summaries())

import os.path
from typing import Any, Dict, NamedTuple

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


class _TrainerState(NamedTuple):
    model: NestedParameters
    learner: learner.LearnerState


class SpmdTrainer(Module):

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("dir", None, "The root directory to store checkpoints and summaries.")
        cfg.define("input", None, "The model input config.")
        cfg.define("model", None, "The model config.")
        cfg.define("learner", None, "The learner config.")
        cfg.define("summary_writer", summary_writer.SummaryWriter.default_config(), "The summary writer.")
        cfg.define("checkpointer", checkpointer.Checkpointer.default_config(), "The checkpointer.")
        return cfg

    def __init__(self, cfg: config_lib.Config):
        super().__init__(cfg, parent=None)
        cfg = self.config
        # self._add_child('input', cfg.input)
        self.input = cfg.input.instantiate()
        self._add_child('model', cfg.model)
        self._add_child('learner', cfg.learner)
        self._add_child('summary_writer', cfg.summary_writer.set(dir=os.path.join(cfg.dir, "summaries")))
        self._add_child('checkpointer', cfg.checkpointer.set(dir=os.path.join(cfg.dir, "checkpoints")))

    def init(self, prng_key: jax.random.KeyArray):
        def init_state(prng_key: jax.random.KeyArray):
            model_params = self.model.initialize_parameters_recursively(prng_key)
            learner_params = self.learner.init(model_params)
            return _TrainerState(model=model_params, learner=learner_params)

        # TODO(ruoming): support state sharding.
        logging.info("Compiling init_state")
        init_computation = self._pjit(init_state, in_axis_resources=(None,), out_axis_resources=None)
        logging.info("Initializing states")
        self._state = init_computation(prng_key)
        flat_paths, _ = jax.tree_flatten(tree_paths(self._state))
        flat_values, _ = jax.tree_flatten(self._state)
        for path, value in zip(flat_paths, flat_values):
            logging.info("Trainer state: %s=%s(%s)", path, value.dtype, value.shape)
        logging.info("Compiling computation")
        self._computation = self._pjit(self._forward_and_update,
                                       in_axis_resources=(None,  # prng_key
                                           None,  # state
                                           self._input_sharding(),  # input_batch
                                           ),
                                       out_axis_resources=dict(state=None, summaries=None, loss=None, aux=None))
        logging.info("Compiling computation done")

    def run_step(self, step: int, prng_key: jax.random.KeyArray):
        self.checkpointer.save(step, self._state)
        input_batch = next(self.input)
        # Note(Jan 2022): pjit currently requires all parameters to be specified as positional args.
        outputs = self._computation(prng_key, self._state, input_batch)
        logging.info("Process % 3d step % 8d: loss=%s aux=%s",
                     jax.process_index(), step, outputs["loss"], outputs["aux"])
        self._state = outputs["state"]
        self.summary_writer(step, outputs["summaries"])

    def _pjit(self, fn, **kwargs):
        if not all(device.platform in ('tpu', 'gpu') for device in jax.devices()):
            logging.info('Falling back to jit on %s', [device.platform for device in jax.devices()])
            return jax.jit(fn)
        return pjit(fn, **kwargs)

    def _forward_and_update(self, prng_key: jax.random.KeyArray, state: module.NestedParameters,
                            input_batch: Dict[str, Any]):
        forward_key, learner_key = jax.random.split(prng_key)
        del prng_key

        def _forward(model_parameters, input_batch):
            forward_context: InvocationContext = self.model.make_invocation_context(
                parameters=model_parameters, is_training=True, prng_key=forward_key)
            with module.root_context(forward_context):
                loss, aux = self.model(input_batch['image'], input_batch['label'])
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

    def _input_sharding(self):
        # Shard data along the batch dim.
        return PartitionSpec('data')

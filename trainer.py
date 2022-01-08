from typing import Any, Dict

import jax
from absl import logging
from jax import numpy as jnp
from jax.experimental import PartitionSpec
from jax.experimental.pjit import pjit

import config as config_lib
import module
from module import Module


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


class SpmdTrainer(Module):

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("input", None, "The model input config.")
        cfg.define("model", None, "The model config.")
        cfg.define("learner", None, "The learner config.")
        cfg.define("summary_writer", None, "The summary writer.")
        cfg.define("checkpointer", None, "The checkpointer.")
        return cfg

    def __init__(self, cfg: config_lib.Config):
        super().__init__(cfg)
        cfg = self.config
        self._add_child('input', cfg.input)
        self._add_child('model', cfg.model)
        self._add_child('learner', cfg.learner)
        self._add_child('summary_writer', cfg.summary_writer)
        self._add_child('checkpointer', cfg.checkpointer)

    def init(self, prng_key: jax.random.KeyArray):
        def init_params(prng_key: jax.random.KeyArray):
            model_params = self.model.initialize_parameters_recursively(prng_key)
            learner_params = self.learner.init(model_params)
            return dict(model=model_params, learner=learner_params)

        self.parameters = pjit(init_params, in_axis_resources=(None,), out_axis_resources=self._parameter_sharding())(
            prng_key)

    def run_step(self, step: int, prng_key: jax.random.KeyArray):
        self.checkpointer.update(step, self.parameters)
        input_batch = next(self.input)
        computation = pjit(self._forward_and_update,
                           in_axis_resources=dict(prng_key=None, parameters=self._parameter_sharding(),
                                                  input_batch=self._input_sharding()),
                           out_axis_resources=dict(parameters=self._parameter_sharding(), summaries=None))
        outputs = computation(prng_key=prng_key, parameters=self.parameters, input_batch=input_batch)
        logging.info("Process % 3d step % 8d: loss=%s aux=%s",
                     jax.process_index(), step, outputs["loss"], outputs["aux"])
        self.parameters = outputs["parameters"]
        self.summary_writer(step, outputs["summaries"])

    def _forward_and_update(self, prng_key: jax.random.KeyArray, parameters: module.NestedParameters,
                            input_batch: Dict[str, Any]):
        context = self.make_invocation_context(parameters=parameters, is_training=True, prng_key=prng_key)
        with module.root_context(context):
            _forward_and_grad = jax.value_and_grad(self.model, has_aux=True)
            (loss, model_aux_outputs), grads = _forward_and_grad(**input_batch)
            updated_model_params = self.learner(method="update", model_params=parameters["model"], gradients=grads)
        updated_parameters = _apply_updates(dict(model=updated_model_params), context.get_parameter_updates())
        summaries = context.get_summaries()
        return dict(parameters=updated_parameters, summaries=summaries, loss=loss, aux=model_aux_outputs)

    def _parameter_sharding(self):
        return None

    def _input_sharding(self):
        # Shard data along the batch dim.
        return PartitionSpec('data')

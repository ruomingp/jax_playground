import os.path
from typing import Any, Dict

import jax
from absl import logging
from jax import numpy as jnp
from jax.experimental import PartitionSpec
from jax.experimental.pjit import pjit

import checkpointer
import config as config_lib
import module
import summary_writer
from module import InvocationContext, Module, tree_paths


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
        def init_params(prng_key: jax.random.KeyArray):
            model_params = self.model.initialize_parameters_recursively(prng_key)
            learner_params = self.learner.init(model_params)
            return dict(model=model_params, learner=learner_params)

        self._parameters = self._pjit(init_params, in_axis_resources=(None,),
                                      out_axis_resources=self._parameter_sharding())(
            prng_key)
        flat_params, _ = jax.tree_flatten(tree_paths(self._parameters))
        for entry in flat_params:
            logging.info("Trainer state: %s", entry)
        # logging.info("Trainer parameters = %s", jax.tree_map(lambda x: x.shape, self._parameters))

    def run_step(self, step: int, prng_key: jax.random.KeyArray):
        self.checkpointer.save(step, self._parameters)
        input_batch = next(self.input)
        computation = self._pjit(self._forward_and_update,
                                 in_axis_resources=dict(prng_key=None, parameters=self._parameter_sharding(),
                                                        input_batch=self._input_sharding()),
                                 out_axis_resources=dict(parameters=self._parameter_sharding(), summaries=None))
        outputs = computation(prng_key=prng_key, parameters=self._parameters, input_batch=input_batch)
        logging.info("Process % 3d step % 8d: loss=%s aux=%s",
                     jax.process_index(), step, outputs["loss"], outputs["aux"])
        self._parameters = outputs["parameters"]
        self.summary_writer(step, outputs["summaries"])

    def _pjit(self, fn, **kwargs):
        if not all(device.platform in ('tpu', 'gpu') for device in jax.devices()):
            logging.info('Skipping pjit on devices %s', jax.devices())
            return fn
        return pjit(fn, **kwargs)

    def _forward_and_update(self, prng_key: jax.random.KeyArray, parameters: module.NestedParameters,
                            input_batch: Dict[str, Any]):
        prng_key, forward_key, learner_key = jax.random.split(prng_key, 3)

        def _forward(model_parameters, input_batch):
            forward_context: InvocationContext = self.model.make_invocation_context(
                parameters=model_parameters, is_training=True, prng_key=forward_key)
            with module.root_context(forward_context):
                loss, aux = self.model(**input_batch)
            return loss, dict(aux=aux, parameter_updates=forward_context.get_parameter_updates(),
                              summaries=forward_context.get_summaries())

        _forward_and_grad = jax.value_and_grad(_forward, has_aux=True)
        (loss, forward_outputs), grads = _forward_and_grad(parameters['model'],
                                                           jax.tree_map(lambda x: jnp.asarray(x), input_batch))

        learner_context: InvocationContext = self.learner.make_invocation_context(
            parameters=parameters["learner"], is_training=True, prng_key=learner_key)
        with module.root_context(learner_context):
            updated_model_params = self.learner(method="update", model_params=parameters["model"], gradients=grads)
        updated_parameters = dict(
            model=_apply_updates(updated_model_params, forward_outputs["parameter_updates"]),
            learner=learner_context.get_parameter_updates())
        summaries = dict(model=forward_outputs["summaries"], learner=learner_context.get_summaries())
        return dict(parameters=updated_parameters, summaries=summaries, loss=loss, aux=forward_outputs["aux"])

    def _parameter_sharding(self):
        return None

    def _input_sharding(self):
        # Shard data along the batch dim.
        return PartitionSpec('data')

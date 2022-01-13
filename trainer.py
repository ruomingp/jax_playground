from typing import Any, Callable, Dict, NamedTuple, Optional

import jax
from absl import logging
from jax import numpy as jnp
from jax.experimental import PartitionSpec
from jax.experimental.pjit import pjit
from functools import partial

import checkpointer
import config
import config as config_lib
import learner
import metrics
import module
import summary_writer
from module import InvocationContext, Module, NestedTensor
from utils import tree_paths, flatten_items


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


class _SpmdRunner(Module):
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
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("input", cfg.input)
        self._add_child("summary_writer", cfg.summary_writer)
        self._state = None
        self._jit_compute = None

    def _jit(self, fn: Callable, **kwargs):
        logging.debug("Compiling computation %s", fn)
        if all(device.platform in ("tpu", "gpu") for device in jax.devices()):
            fn = pjit(fn, **kwargs)
        else:
            logging.log_first_n(
                logging.INFO,
                "Falling back to jit on %s",
                1,
                [device.platform for device in jax.devices()],
            )
            fn = jax.jit(fn)
        logging.debug("Compiling computation done")
        return fn

    def _parameter_sharding(self):
        # TODO(ruoming): support state sharding.
        return None

    def _state_sharding(self):
        return _TrainerState(
            model=self._parameter_sharding(), learner=self._parameter_sharding()
        )

    def _input_sharding(self):
        # Shard data along the batch dim.
        return PartitionSpec("data")


class _TrainerState(NamedTuple):
    model: NestedTensor
    learner: learner.LearnerState


class SpmdTrainer(_SpmdRunner):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
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
        self._add_child("model", cfg.model)
        self._add_child("learner", cfg.learner)
        self._add_child("checkpointer", cfg.checkpointer)
        for evaler_cfg in cfg.evalers:
            self._add_child(evaler_cfg.name, evaler_cfg)

        self._model_param_specs = self.model.create_parameter_specs_recursively()
        logging.info("Model param specs: %s", self._model_param_specs)
        model_param_partition_specs = jax.tree_map(
            lambda spec: PartitionSpec(*spec.partition_spec), self._model_param_specs
        )
        # for path, spec in flatten_items(model_param_partition_specs):
        #     logging.info("Model param partition: %s=%s", path, spec)
        logging.info("Model param partition: %s", model_param_partition_specs)
        learner_state_partition_specs = self.learner.create_state_partition_specs(
            model_param_partition_specs
        )
        self._trainer_state_partition_specs = _TrainerState(
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
        )

    def run(self, prng_key: jax.random.KeyArray, max_step: int):
        prng_key, init_key = jax.random.split(prng_key)
        self._init(init_key)

        for step in range(1, max_step + 1):
            prng_key, step_key = jax.random.split(prng_key)
            self._run_step(step, step_key)

    def _init(self, prng_key: jax.random.KeyArray):
        def _init_state(prng_key: jax.random.KeyArray):
            model_params = self.model.initialize_parameters_recursively(
                prng_key, self._model_param_specs
            )
            learner_params = self.learner.init(model_params)
            return _TrainerState(model=model_params, learner=learner_params)

        init_computation = self._jit(
            _init_state,
            in_axis_resources=(None,),
            out_axis_resources=self._trainer_state_partition_specs,
        )
        logging.info("Initializing states")
        self._state = init_computation(prng_key)
        flat_paths, _ = jax.tree_flatten(tree_paths(self._state))
        flat_values, _ = jax.tree_flatten(self._state)
        for path, value in zip(flat_paths, flat_values):
            logging.info(
                "%s state: %s=%s(%s)", self.path(), path, value.dtype, value.shape
            )

        # Try to restore the latest checkpoint.
        try:
            self._state = self.checkpointer.restore(step=None, state=self._state)
        except Exception as e:
            logging.info("Failed to restore checkpoint: %s", e)
            self.checkpointer.save(step=0, state=self._state)

    def _run_step(self, step: int, prng_key: jax.random.KeyArray):
        cfg = self.config
        input_batch = next(self.input)
        prng_key, train_key = jax.random.split(prng_key)
        # Note(Jan 2022): pjit currently requires all parameters to be specified as positional args.
        outputs = self._jit_train_step(train_key, self._state, input_batch)
        self._state = outputs["state"]
        logging.info(
            "Process % 3d step % 8d: loss=%s aux=%s",
            jax.process_index(),
            step,
            outputs["loss"],
            outputs["aux"],
        )
        self.summary_writer(step, {"loss": outputs["loss"], **outputs["summaries"]})
        for evaler_cfg in cfg.evalers:
            prng_key, eval_key = jax.random.split(prng_key)
            self._children[evaler_cfg.name].run_step(
                step, eval_key, model=self.model, model_params=self._state.model
            )
        self.checkpointer.save(step=step, state=self._state)

    def _train_step(
            self,
            prng_key: jax.random.KeyArray,
            state: _TrainerState,
            input_batch: Dict[str, Any],
    ):
        prng_key, forward_key, learner_key = jax.random.split(prng_key, 3)

        def _forward(model_parameters, input_batch):
            (loss, aux), model_output_collection = module.functional(
                self.model, state=model_parameters, is_training=True, prng_key=forward_key, inputs=input_batch,
                output_collection_sections=("summaries", "state_updates"))
            return loss, dict(aux=aux, **model_output_collection)

        _forward_and_grad = jax.value_and_grad(_forward, has_aux=True)
        (loss, forward_output_collection), grads = _forward_and_grad(
            state.model, jax.tree_map(lambda x: jnp.asarray(x), input_batch)
        )

        (updated_learner_state, updated_model_params), learner_output_collection = module.functional(
            self.learner, method="update",
            state=None, is_training=True, prng_key=learner_key,
            inputs=dict(state=state.learner, model_params=state.model, gradients=grads),
            output_collection_sections=["summaries"])
        updated_state = _TrainerState(
            model=_apply_updates(
                updated_model_params, forward_output_collection["state_updates"]
            ),
            learner=updated_learner_state,
        )
        # TODO(ruoming): only retrieve summaries when necessary.
        summaries = dict(
            model=forward_output_collection["summaries"], learner=learner_output_collection["summaries"],
        )
        return dict(
            state=updated_state,
            summaries=summaries,
            loss=loss,
            aux=forward_output_collection["aux"],
        )


class SpmdEvaler(_SpmdRunner):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("run_every_n_steps", 1, "Run this evaler every N steps.")
        cfg.define(
            "metric_accumulator",
            metrics.MetricAccumulator.default_config(),
            "The eval metric accumulator config.",
        )
        return cfg

    def run_step(
            self,
            step: int,
            prng_key: jax.random.KeyArray,
            *,
            model: Module,
            model_params: NestedTensor
    ):
        cfg = self.config
        if step % cfg.run_every_n_steps != 0:
            return
        _jit_eval_batch = self._jit(
            partial(module.functional, model, is_training=False, output_collection_sections=["summaries"]),
            in_axis_resources=(
                None,  # prng_key
                self._parameter_sharding(),
                self._input_sharding(),
            ),
            out_axis_resources=None,
        )
        prng_key, init_key = jax.random.split(prng_key)
        metric_accumulator = cfg.metric_accumulator.instantiate()
        num_batches = 0
        for input_batch in self.input:
            prng_key, batch_key = jax.random.split(prng_key)
            (_, aux), output_collection = _jit_eval_batch(batch_key, model_params, input_batch)
            if num_batches == 0:
                self.summary_writer(step, output_collection["summaries"])
            num_batches += 1
            metric_accumulator.update(input_batch, aux)
        summaries = metric_accumulator.summaries()
        logging.info(
            "Process % 3d step % 8d: %s.metrics=%s",
            jax.process_index(),
            step,
            self.path(),
            summaries,
        )
        self.summary_writer(step, summaries)

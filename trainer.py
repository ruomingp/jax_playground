import math
import os.path
import time
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Union

import jax
from absl import logging
from jax import numpy as jnp
from jax.experimental import PartitionSpec
from jax.experimental import maps
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit

import checkpointer
import config as config_lib
import learner
import metrics
import summary_writer
from module import functional as F, Module, NestedTensor, NestedPartitionSpec
from utils import Tensor, flatten_items, tree_paths


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

    def _jit(self, fn: Callable, *, in_axis_resources, out_axis_resources, **kwargs):
        self.vlog(3, "Compiling computation %s", fn)
        if all(device.platform in ("tpu", "gpu") for device in jax.devices()):
            fn = pjit(
                fn,
                in_axis_resources=in_axis_resources,
                out_axis_resources=out_axis_resources,
                **kwargs,
            )
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


class _TrainerState(NamedTuple):
    step: Union[Tensor, NestedPartitionSpec]
    prng_key: Union[jax.random.KeyArray, NestedPartitionSpec]
    model: Union[NestedTensor, NestedPartitionSpec]
    learner: Union[learner.LearnerState, NestedPartitionSpec]


class SpmdTrainer(_SpmdRunner):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("dir", None,
                   "(Required) "
                   "The trainer root dir. "
                   "By default, checkpoints will be written under {dir}/checkpoints/ "
                   "and summaries will be written under {dir}/summaries.")
        cfg.define("max_step", math.inf, "The maximum number of steps.")
        cfg.define("mesh_shape", None,
                   "The device mesh shape in the form of a tuple of ints. "
                   "Must have the same length as mesh_axis_names. "
                   "Defaults to (jax.device_count(), 1), "
                   "which represents a data-parallel mesh when the mesh axis names are ('data', 'model').")
        cfg.define("mesh_axis_names", ("data", "model"),
                   "The mesh axis names. The names can be referenced in ParameterSpec.partition_spec.")
        cfg.define("model", None, "The model config.")
        cfg.define("learner", None, "The learner config.")
        cfg.define(
            "checkpointer",
            checkpointer.Checkpointer.default_config(),
            "The checkpointer config.",
        )
        cfg.define(
            "evalers",
            dict(),
            "A dict of evaler names to configs, each name must be non-empty.",
        )
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        cfg.summary_writer.dir = cfg.summary_writer.dir or os.path.join(cfg.dir, "summaries", "train_train")
        super().__init__(cfg, parent=parent)
        cfg = self.config

        self._add_child("model", cfg.model)
        self._add_child("learner", cfg.learner)

        cfg.checkpointer.dir = cfg.checkpointer.dir or os.path.join(cfg.dir, "checkpoints")
        self._add_child("checkpointer", cfg.checkpointer)

        for evaler_name, evaler_cfg in cfg.evalers.items():
            evaler_cfg.summary_writer.dir = evaler_cfg.summary_writer.dir or os.path.join(cfg.dir, "summaries",
                                                                                          evaler_name)
            self._add_child(evaler_name, evaler_cfg)

        self._model_param_specs = self.model.create_parameter_specs_recursively()
        self.vlog(3, "Model param specs: %s", self._model_param_specs)
        model_param_partition_specs = jax.tree_map(
            lambda spec: PartitionSpec(*spec.partition_spec), self._model_param_specs
        )
        for path, spec in flatten_items(model_param_partition_specs):
            self.vlog(3, "Model param partition: %s=%s", path, spec)
        self._step_log("Model param partition: %s", model_param_partition_specs)
        learner_state_partition_specs = self.learner.create_state_partition_specs(
            model_param_partition_specs
        )
        self._trainer_state_partition_specs = _TrainerState(
            step=None,
            prng_key=None,
            model=model_param_partition_specs,
            learner=learner_state_partition_specs,
        )
        self._jit_train_step = self._jit(
            self._train_step,
            in_axis_resources=(
                self._trainer_state_partition_specs,
                self._input_sharding(),
            ),
            out_axis_resources=None,
            donate_argnums=(0,),  # donate the state
        )
        for evaler_name in cfg.evalers:
            self._children[evaler_name].init(
                self.model, model_param_partition_specs
            )

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
            **kwargs,
        )

    def run(self, prng_key: jax.random.KeyArray) -> Optional[NestedTensor]:
        cfg = self.config
        jax.config.update("jax_log_compiles", True)
        mesh_shape = cfg.mesh_shape or (jax.device_count(), 1)
        devices = mesh_utils.create_device_mesh(mesh_shape)
        mesh = maps.Mesh(devices, cfg.mesh_axis_names)
        with maps.mesh(mesh.devices, mesh.axis_names):
            self._init(prng_key)

            if self.step >= cfg.max_step:
                self._step_log("Already reached max_step=%s. Stopping", cfg.max_step)
                return None

            start_time = time.perf_counter()
            num_steps = 0
            output = None
            for input_batch in self.input:
                self.vlog(3, "Start step %s", self.step + 1)
                output = self._run_step(input_batch)
                self.vlog(3, "Done step %s", self.step)
                num_steps += 1
                if num_steps % 100 == 0:
                    now = time.perf_counter()
                    self._step_log(
                        "Average step time: %s seconds", (now - start_time) / num_steps
                    )
                    num_steps = 0
                    start_time = now
                if self.step >= cfg.max_step:
                    self._step_log("Reached max_step=%s. Stopping", cfg.max_step)
                    return output
            self._step_log("Reached end of inputs. Stopping")
            return output

    def _init(self, prng_key: jax.random.KeyArray):
        def _init_state(prng_key: jax.random.KeyArray):
            prng_key, init_key = jax.random.split(prng_key)
            model_params = self.model.initialize_parameters_recursively(
                init_key, self._model_param_specs
            )
            learner_params = self.learner.init(model_params)
            return _TrainerState(
                step=jnp.zeros([], dtype=jnp.int64),
                prng_key=prng_key,
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

    def _run_step(self, input_batch: NestedTensor) -> NestedTensor:
        """Runs a single training step.

        Args:
            input_batch: a NestedTensor.

        Returns:
            A dict containing 'loss' and 'aux' outputs.
        """
        cfg = self.config
        self.vlog(3, "  train_step: %s", self.step + 1)
        # with jax.profiler.trace(cfg.summary_writer.dir):
        with jax.profiler.StepTraceAnnotation("train", step_num=self.step):
            # Note(Jan 2022): pjit currently requires all parameters to be specified as positional args.
            outputs = self._jit_train_step(self._state, input_batch)
            self._state = outputs["state"]
        if self._state.step % 100 == 0:
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
        # Note: we will use the same eval key as the training keys of the future step, which should be okay.
        prng_key = self._state.prng_key
        for evaler_name in cfg.evalers:
            prng_key, eval_key = jax.random.split(prng_key)
            self._children[evaler_name].eval_step(
                self.step,
                prng_key=eval_key,
                model_params=self._state.model,
            )
        self.vlog(3, "  checkpointer: %s", self.step)
        self.checkpointer.save(step=self.step, state=self._state)
        return {"loss": outputs["loss"], "aux": outputs["aux"]}

    def _train_step(
            self,
            state: _TrainerState,
            input_batch: Dict[str, Any],
    ):
        new_prng_key, forward_key, learner_key = jax.random.split(state.prng_key, 3)

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
            prng_key=new_prng_key,
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

    def init(self, model: Module, model_param_partition_spec: NestedPartitionSpec):
        self._jit_eval_batch = self._jit(
            partial(
                F,
                model,
                is_training=False,
            ),
            in_axis_resources=(
                None,  # prng_key
                model_param_partition_spec,
                self._input_sharding(),
            ),
            out_axis_resources=None,
        )

    def eval_step(
            self,
            step: int,
            *,
            prng_key: jax.random.KeyArray,
            model_params: NestedTensor,
    ):
        cfg = self.config
        if step % cfg.run_every_n_steps != 0:
            return
        self.vlog(3, "%s start at %s", self.path(), step)
        prng_key, init_key = jax.random.split(prng_key)
        metric_accumulator = cfg.metric_accumulator.instantiate()
        num_batches = 0
        for input_batch in self.input:
            prng_key, batch_key = jax.random.split(prng_key)
            with jax.profiler.StepTraceAnnotation(cfg.name, step_num=step):
                (_, aux), output_collection = self._jit_eval_batch(
                    batch_key, model_params, input_batch
                )
            num_batches += 1
            self.vlog(
                3,
                "Process % 3d step % 8d batch % 8d: %s.aux=%s",
                jax.process_index(),
                step,
                num_batches,
                self.path(),
                jax.tree_map(lambda x: x.item() if x.ndim == 0 else f"T{x.shape}", aux),
            )
            metric_accumulator.update(output_collection.summaries)
        summaries = metric_accumulator.summaries()
        self.vlog(
            2,
            "Process % 3d step % 8d: %s.metrics=%s",
            jax.process_index(),
            step,
            self.path(),
            summaries,
        )
        self.summary_writer(step, summaries)
        self.vlog(3, "%s done at %s", self.path(), step)

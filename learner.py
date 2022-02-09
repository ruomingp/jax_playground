"""Optimization modules."""
import copy
from typing import Callable, NamedTuple, Optional

import jax
import optax
from absl import logging
from jax import numpy as jnp

import config as config_lib
import schedule
import utils
from module import Module, NestedPartitionSpec, NestedTensor, Tensor, current_context

TransformPartitionSpecFn = Callable[[NestedPartitionSpec], NestedPartitionSpec]


class PartitionedGradientTransformation(NamedTuple):
    init: optax.TransformInitFn
    update: optax.TransformUpdateFn
    partition: TransformPartitionSpecFn


def chain(*args):
    def to_partitioned_transformation(transformation):
        if isinstance(transformation, (config_lib.InstantiableConfig, config_lib.FunctionConfig)):
            transformation = transformation.instantiate()
        if isinstance(transformation, optax.GradientTransformation):
            transformation = replicate(transformation)
        if not isinstance(transformation, PartitionedGradientTransformation):
            raise ValueError(
                f"Expected PartitionedGradientTransformation. Got {type(transformation)}: {transformation}"
            )
        return transformation

    args = [to_partitioned_transformation(e) for e in args]

    base = optax.chain(*[optax.GradientTransformation(init=e.init, update=e.update) for e in args])

    def partition(input_partition_spec):
        return tuple(e.partition(input_partition_spec) for e in args)

    return PartitionedGradientTransformation(
        init=base.init, update=base.update, partition=partition
    )


def copy_partition(
    base: optax.GradientTransformation,
) -> PartitionedGradientTransformation:
    return PartitionedGradientTransformation(
        init=base.init,
        update=base.update,
        partition=lambda partition_spec: copy.deepcopy(partition_spec),
    )


def trace_partition(
    base: optax.GradientTransformation,
) -> PartitionedGradientTransformation:
    return PartitionedGradientTransformation(
        init=base.init,
        update=base.update,
        partition=lambda partition_spec: optax.TraceState(trace=partition_spec),
    )


def replicate(base: optax.GradientTransformation) -> PartitionedGradientTransformation:
    return PartitionedGradientTransformation(
        init=base.init, update=base.update, partition=lambda partition_spec: None
    )


def scale_from_learning_rate(learning_rate: schedule.Schedule):
    learning_rate_fn = schedule.as_schedule_fn(learning_rate)

    def scale(step):
        lr = learning_rate_fn(step)
        context = current_context()
        if context:
            context.add_summary("lr_schedule_step", step)
            context.add_summary("learning_rate", lr)
        return -lr

    return scale


def sgd_optimizer(
    learning_rate: schedule.Schedule,
    momentum: float = 0,
    weight_decay: float = 0,
) -> PartitionedGradientTransformation:
    return chain(
        trace_partition(optax.trace(decay=momentum)),
        replicate(optax.add_decayed_weights(weight_decay)),
        replicate(optax.scale_by_schedule(scale_from_learning_rate(learning_rate))),
    )


def clip_by_global_norm(
    max_norm: Optional[float] = None, *, eps: float = 1e-8
) -> PartitionedGradientTransformation:
    """Clips gradeints s.t. global norm <= max_norm."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        g_norm = optax.global_norm(updates)
        context = current_context()
        if context is not None:
            context.add_summary("gradient_norm", g_norm)
        if max_norm is not None:
            g_scale = jnp.minimum(1.0, max_norm / (g_norm + eps))
            if context is not None:
                context.add_summary("gradient_scale", g_scale)
            updates = jax.tree_map(lambda t: t * g_scale, updates)
        return updates, state

    return PartitionedGradientTransformation(
        init=init_fn, update=update_fn, partition=lambda partition_spec: None
    )


class LearnerState(NamedTuple):
    optimizer: optax.OptState


class Learner(Module):
    """The learner module."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("optimizer", None, "The optimizer config.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self.optimizer: PartitionedGradientTransformation = cfg.optimizer.instantiate()
        if not isinstance(self.optimizer, PartitionedGradientTransformation):
            raise ValueError(
                f"optimizer must be a PartitionedGradientTransformation: {cfg.optimizer}"
            )

    def create_state_partition_specs(self, model_param_partition_specs: NestedPartitionSpec):
        return LearnerState(optimizer=self.optimizer.partition(model_param_partition_specs))

    def init(self, model_params: NestedTensor) -> LearnerState:
        return LearnerState(optimizer=self.optimizer.init(model_params))

    def update(self, *, gradients: NestedTensor, model_params: NestedTensor) -> NestedTensor:
        """Computes `model_params` updates with `gradients`."""
        parameter_updates, optimizer_state = self.optimizer.update(
            gradients, state=self.state.optimizer, params=model_params
        )
        self.add_state_update("optimizer", optimizer_state)
        logging.info("model_params=%s updates=%s", model_params, parameter_updates)
        updated_model_params = optax.apply_updates(model_params, parameter_updates)
        return updated_model_params

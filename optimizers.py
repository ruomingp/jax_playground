"""Optimization modules."""
from typing import Any, Optional, Tuple

import jax
import optax
from jax import numpy as jnp

import config as config_lib
import schedule
from factorized_rms import scale_by_factored_rms
from module import NestedParameterPartitionSpec, NestedPartitionSpec, NestedTensor, current_context
from optimizer_base import (
    NestedOptParam,
    OptStatePartitionSpec,
    PartitionedGradientTransformation,
    TransformPartitionSpecFn,
)
from utils import vectorized_tree_map


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

    def partition(param_partition_spec):
        return tuple(e.partition(param_partition_spec) for e in args)

    return PartitionedGradientTransformation(
        init=base.init, update=base.update, partition=partition
    )


def with_partition_fn(
    base: optax.GradientTransformation, partition_fn: TransformPartitionSpecFn
) -> PartitionedGradientTransformation:
    def param_values(params):
        return jax.tree_map(lambda opt_param: opt_param.value, params)

    def init_fn(params: NestedOptParam) -> NestedTensor:
        return base.init(param_values(params))

    def update_fn(
        updates: optax.Updates, state: optax.OptState, params: NestedOptParam
    ) -> Tuple[optax.Updates, optax.OptState]:
        return base.update(updates, state, param_values(params))

    return PartitionedGradientTransformation(init=init_fn, update=update_fn, partition=partition_fn)


def replicate(base: optax.GradientTransformation) -> PartitionedGradientTransformation:
    return with_partition_fn(
        base,
        lambda param_partition_specs: OptStatePartitionSpec(shape=None, partition=None),
    )


def copy_partition(param_partition_specs: NestedParameterPartitionSpec) -> NestedPartitionSpec:
    return jax.tree_map(
        lambda pps: OptStatePartitionSpec(shape=pps.shape, partition=pps.partition),
        param_partition_specs,
    )


def trace_partition(
    base: optax.GradientTransformation,
) -> PartitionedGradientTransformation:
    def partition_fn(param_partition_specs: NestedParameterPartitionSpec) -> NestedPartitionSpec:
        return optax.TraceState(trace=copy_partition(param_partition_specs))

    return with_partition_fn(base, partition_fn)


def ema_partition(
    base: optax.GradientTransformation,
) -> PartitionedGradientTransformation:
    def partition_fn(param_partition_specs: NestedParameterPartitionSpec) -> NestedPartitionSpec:
        return optax.EmaState(count=None, ema=copy_partition(param_partition_specs))

    return with_partition_fn(base, partition_fn)


def adam_partition(base: optax.GradientTransformation) -> PartitionedGradientTransformation:
    def partition_fn(param_partition_specs: NestedParameterPartitionSpec) -> NestedPartitionSpec:
        return optax.ScaleByAdamState(
            count=None,
            mu=copy_partition(param_partition_specs),
            nu=copy_partition(param_partition_specs),
        )

    return with_partition_fn(base, partition_fn)


def scale_from_learning_rate(learning_rate: schedule.Schedule, *, flip_sign=True):
    learning_rate_fn = schedule.as_schedule_fn(learning_rate)

    def scale(step):
        lr = learning_rate_fn(step)
        context = current_context()
        if context:
            context.add_summary("lr_schedule_step", step)
            context.add_summary("learning_rate", lr)
        return -lr if flip_sign else lr

    return scale


def sgd_optimizer(
    learning_rate: schedule.Schedule,
    momentum: float = 0,
    weight_decay: float = 0,
) -> PartitionedGradientTransformation:
    return chain(
        trace_partition(optax.trace(decay=momentum)),
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_schedule(scale_from_learning_rate(learning_rate)),
    )


def adamw_optimizer(
    learning_rate: schedule.Schedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-4,
    mu_dtype: Optional[jnp.dtype] = None,
):
    return chain(
        adam_partition(optax.scale_by_adam(b1=b1, b2=b2, eps=eps, mu_dtype=mu_dtype)),
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_schedule(scale_from_learning_rate(learning_rate)),
    )


def adafactor_optimizer(
    learning_rate: schedule.Schedule,
    decay_rate: float = 0.8,
    decay_offset: int = 0,
    multiply_by_parameter_scale: float = True,
    clipping_threshold: Optional[float] = 1.0,
    momentum: Optional[float] = None,
    dtype_momentum: Any = jnp.float32,
    weight_decay_rate: Optional[float] = None,
    eps: float = 1e-30,
    factored: bool = True,
):
    tx = [
        scale_by_factored_rms(
            factored, decay_rate=decay_rate, step_offset=decay_offset, epsilon=eps
        )
    ]
    # This basic rescaling is typically combined with one or more of the following
    # transformation (all can be disabled via adafactor's constructor args).
    if clipping_threshold is not None:
        tx.append(clip_by_block_rms(clipping_threshold))
    if learning_rate is not None:
        tx.append(optax.scale_by_schedule(scale_from_learning_rate(learning_rate, flip_sign=False)))
    if multiply_by_parameter_scale:
        tx.append(scale_by_param_block_rms())
    if momentum is not None:
        tx.append(
            ema_partition(optax.ema(momentum, debias=False, accumulator_dtype=dtype_momentum))
        )
    if weight_decay_rate is not None:
        tx.append(optax.add_decayed_weights(weight_decay_rate))
    # In gradient "descent" we follow the negative gradient.
    tx.append(optax.scale(-1))
    return chain(*tx)


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


def clip_by_block_rms(threshold: float) -> optax.GradientTransformation:
    """Clip updates to a max rms for the gradient of each param vector or matrix.

    A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
    (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.
    A sub tree under a VDict will be vectorized and clipped separately so that we
    clip updates to different layers of a Repeat/Pipeline layer separately.

    Args:
        threshold: the maximum rms for the gradient of each param vector or matrix.

    Returns:
        An (init_fn, update_fn) tuple.
    """

    def init_fn(_):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params

        def _clip_fn(u):
            clip_denom = jnp.maximum(1.0, jnp.sqrt(jnp.mean(u**2)) / threshold)
            return u / clip_denom

        # The only difference from the optax implementation: vectorized_tree_map vs. jax.tree_map.
        updates = vectorized_tree_map(_clip_fn, updates)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_param_block_rms(min_scale: float = 1e-3) -> optax.GradientTransformation:
    """Scale updates by rms of the gradient for each param vector or matrix.

    A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
    (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.
    A sub tree under a VDict will be vectorized and scaled separately so that we
    scale updates to different layers of a Repeat/Pipeline layer separately.

    Args:
        min_scale: minimum scaling factor.

    Returns:
        An (init_fn, update_fn) tuple.
    """

    def init_fn(_):
        return optax.EmptyState()

    def update_fn(updates, state, params):
        updates = vectorized_tree_map(
            lambda u, p: u * optax.safe_root_mean_squares(p, min_scale), updates, params
        )
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)

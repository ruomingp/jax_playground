"""Factorized RMS.

Adapted from optax factorized.py.
"""

import dataclasses
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax import FactoredState

from module import (
    FactorizationSpec,
    NestedParameterPartitionSpec,
    NestedPartitionSpec,
    ParameterPartitionSpec,
    PartitionSpec,
)
from optimizer_base import OptStatePartitionSpec, PartitionedGradientTransformation
from utils import Tensor


def _factored_dims(
    factored: bool,
    factorization_spec: Optional[FactorizationSpec],
) -> Optional[Tuple[int, int]]:
    """Whether to use a factored second moment estimator.

    This function returns a tuple with the two axes to reduce over or None.

    Args:
        factored: whether to use factored second-moment estimator for 2d vars.
        factorization_spec: the factorization spec.

    Returns:
        None or a tuple of ints representing (col_axis, row_axis).
    """
    if not factored or factorization_spec is None:
        return None
    row_axes = [
        index for index, axis_name in enumerate(factorization_spec.axes) if axis_name == "row"
    ]
    col_axes = [
        index for index, axis_name in enumerate(factorization_spec.axes) if axis_name == "col"
    ]
    if not row_axes and not col_axes:
        return None
    if len(row_axes) != 1 or len(col_axes) != 1:
        raise ValueError(f"Invalid factorization_spec: {factorization_spec}")
    return col_axes[0], row_axes[0]


def _decay_rate_pow(i: int, exponent: float = 0.8) -> float:
    """Second-order moment decay schedule."""
    t = jnp.array(i, jnp.float32) + 1.0
    return 1.0 - t ** (-exponent)


@dataclasses.dataclass
class _UpdateResult:
    """Opaque containter that is not traversed by jax.tree_multimap."""

    update: Tensor  # the update to apply to params.
    v_row: Tensor  # used for factored params.
    v_col: Tensor  # used for factored params.
    v: Tensor  # used for params where factoring is skipped.


def scale_by_factored_rms(
    factored: bool = True,
    decay_rate: float = 0.8,
    step_offset: int = 0,
    epsilon: float = 1e-30,
) -> PartitionedGradientTransformation:
    """Scaling by a factored estimate of the gradient rms (as in Adafactor).

    This is a so-called "1+epsilon" scaling algorithms, that is extremely memory
    efficient compared to RMSProp/Adam, and has had wide success when applied to
    large-scale training of attention-based models.

    References:
        [Shazeer et al, 2018](https://arxiv.org/abs/1804.04235)

    Args:
        factored: boolean: whether to use factored second-moment estimates..
        decay_rate: float: controls second-moment exponential decay schedule.
        step_offset: for finetuning, one may set this to the starting step-number
          of the fine tuning phase.
        epsilon: Regularization constant for squared gradient.

    Returns:
        the corresponding `GradientTransformation`.
    """

    def _to_state(count: Tensor, result_tree):
        """Maps from a tree of (factored) values to separate trees of values."""
        return FactoredState(
            count=count,
            v_row=jax.tree_map(lambda o: o.v_row, result_tree),
            v_col=jax.tree_map(lambda o: o.v_col, result_tree),
            v=jax.tree_map(lambda o: o.v, result_tree),
        )

    def init_fn(params):
        """Initialise the optimiser's state."""

        def _init(param):
            shape = param.shape
            factored_dims = _factored_dims(factored, param.factorization_spec)
            if factored_dims is not None:
                d1, d0 = factored_dims
                vr_shape = np.delete(shape, d0)
                vc_shape = np.delete(shape, d1)
                return _UpdateResult(
                    update=jnp.zeros((1,)),
                    v_row=jnp.zeros(vr_shape),
                    v_col=jnp.zeros(vc_shape),
                    v=jnp.zeros((1,)),
                )
            else:
                return _UpdateResult(
                    update=jnp.zeros((1,)),
                    v_row=jnp.zeros((1,)),
                    v_col=jnp.zeros((1,)),
                    v=jnp.zeros(param.shape),
                )

        return _to_state(jnp.zeros([], jnp.int32), jax.tree_map(_init, params))

    def update_fn(grads, state, params):
        """Apply gradient transformation."""
        if params is None:
            raise ValueError("param is None")

        def _update(grad, v_row, v_col, v, param, step):
            grad = grad.astype(jnp.float32)
            decay_rate_t = _decay_rate_pow(step - step_offset, decay_rate)

            # Scaled by factorized second moment statistics.
            new_v_row = jnp.zeros((1,))
            new_v_col = jnp.zeros((1,))
            new_v = jnp.zeros((1,))

            factored_dims = _factored_dims(factored, param.factorization_spec)
            if factored_dims is not None:
                d1, d0 = factored_dims
                grad_sqr = grad * grad + epsilon
                new_v_row = decay_rate_t * v_row + (1.0 - decay_rate_t) * jnp.mean(
                    grad_sqr, axis=d0
                )
                new_v_col = decay_rate_t * v_col + (1.0 - decay_rate_t) * jnp.mean(
                    grad_sqr, axis=d1
                )
                reduced_d1 = d1 - 1 if d1 > d0 else d1
                row_col_mean = jnp.mean(new_v_row, axis=reduced_d1, keepdims=True)
                row_factor = (new_v_row / row_col_mean) ** -0.5
                col_factor = (new_v_col) ** -0.5
                update = (
                    grad
                    * jnp.expand_dims(row_factor, axis=d0)
                    * jnp.expand_dims(col_factor, axis=d1)
                )
            else:
                grad_sqr = grad * grad + epsilon
                new_v = decay_rate_t * v + (1.0 - decay_rate_t) * grad_sqr
                update = grad * (new_v) ** -0.5

            return _UpdateResult(update, new_v_row, new_v_col, new_v)

        # Transform grad and compute new per-parameter stats.
        output = jax.tree_multimap(
            lambda *args: _update(*args, state.count),
            grads,
            state.v_row,
            state.v_col,
            state.v,
            params,
        )

        # Unpack updates / stats and return.
        updates = jax.tree_map(lambda o: o.update, output)
        return updates, _to_state(optax.safe_int32_increment(state.count), output)

    @dataclasses.dataclass
    class VxPartitionSpec:
        v_row: OptStatePartitionSpec
        v_col: OptStatePartitionSpec
        v: OptStatePartitionSpec

    def get_vx_partition_spec(param_partition_spec: ParameterPartitionSpec):
        p_shape = param_partition_spec.shape
        p_partition = param_partition_spec.partition
        factorization_spec = param_partition_spec.factorization
        if (
            not factored
            or factorization_spec is None
            or all(f is None for f in factorization_spec.axes)
        ):
            return VxPartitionSpec(
                v_row=None,
                v_col=None,
                v=OptStatePartitionSpec(shape=p_shape, partition=p_partition),
            )
        factorization = factorization_spec.axes
        vr_partition_spec = OptStatePartitionSpec(
            shape=[dim for dim, f in zip(p_shape, factorization) if f != "row"],
            partition=PartitionSpec(*[p for p, f in zip(p_partition, factorization) if f != "row"]),
        )
        vc_partition_spec = OptStatePartitionSpec(
            shape=[dim for dim, f in zip(p_shape, factorization) if f != "col"],
            partition=PartitionSpec(*[p for p, f in zip(p_partition, factorization) if f != "col"]),
        )
        if (
            len(vr_partition_spec.partition) != len(p_partition) - 1
            or len(vc_partition_spec.partition) != len(p_partition) - 1
        ):
            raise ValueError(
                f"Unexpected factorization: {factorization} for {param_partition_spec}"
            )
        return VxPartitionSpec(v_row=vr_partition_spec, v_col=vc_partition_spec, v=None)

    def partition_fn(param_partition_specs: NestedParameterPartitionSpec) -> NestedPartitionSpec:
        vx_partition_specs = jax.tree_map(get_vx_partition_spec, param_partition_specs)
        return optax.FactoredState(
            count=None,
            v_row=jax.tree_map(lambda vx: vx.v_row, vx_partition_specs),
            v_col=jax.tree_map(lambda vx: vx.v_col, vx_partition_specs),
            v=jax.tree_map(lambda vx: vx.v, vx_partition_specs),
        )

    return PartitionedGradientTransformation(init=init_fn, update=update_fn, partition=partition_fn)

"""A generic pipeline layer.

https://arxiv.org/abs/1811.06965

Adapted from:
https://github.com/tensorflow/lingvo/blob/master/lingvo/jax/layers/pipeline.py

A pipeline layer consists a stack of N identical sub layers, where
  * The variables are stacked across layers. Each stacked variable has shape [N, ...].
  * The inputs are divided into M microbatches and have shape [M, ...].
  * The processing happens in a loop consisting of M+N-1 steps.
    In each step 0 <= t < M+N-1, microbatch 0 <= m < M will be processed by layer (t - m) if 0 <= t - m < N.
    Or, expressed in layer-parallel terms, layers will process microbatch slice [t:t-N:-1] at step t
    (assuming that we pad the microbatches with N - 1 dummy microbatches at both ends).
"""

from typing import NamedTuple, Optional, Tuple

import jax
from jax import numpy as jnp
from jax.experimental.pjit import PartitionSpec

import config as config_lib
from module import (
    BaseLayer,
    FactorizationSpec,
    Module,
    NestedParameterPartitionSpec,
    NestedPartitionSpec,
    NestedTensor,
    ParameterPartitionSpec,
    Tensor,
    child_context,
    current_context,
    new_output_collection,
    set_current_context,
)
from utils import VDict, shapes, with_sharding_constraint


def transpose_to_pipeline_stage_inputs(x, partition_spec: Optional[PartitionSpec] = None):
    """Transposes `x` from the 'layer-major' layout to the 'pipeline-major' layout.

    Args:
        x: of shape [N, M, ...], where x[i, j] represents layerwise inputs for pipeline layer[i] and microbatch[j].
        partition_spec: the partition spec for x.

    Returns:
        x', a tensor of shape [M + N - 1, N, ...], where x'[t, i] represents the layerwise inputs for timestep[t]
            and layer[i]: x'[i + j, i] == x[i, j].
    """
    N, M = x.shape[:2]
    # [N, M + N, ...].
    x = jnp.pad(x, [(0, 0), (0, N)] + [(0, 0)] * (x.ndim - 2))
    # [N * (M + N), ...].
    x = jnp.reshape(x, [-1] + list(x.shape[2:]))
    # [N * (M + N - 1), ...].
    x = x[:-N]
    # [N, M + N - 1, ...].
    x = jnp.reshape(x, [N, M + N - 1] + list(x.shape[1:]))
    # Apply sharding constraints at the first opportunity after reshapes
    # (i.e. when the input is first in the right shape for the constraint again).
    x = with_sharding_constraint(x, partition_spec)
    # [M + N - 1, N, ...].
    x = jnp.transpose(x, [1, 0] + list(range(2, x.ndim)))
    return x


def transpose_from_pipeline_stage_outputs(x, partition_spec: Optional[PartitionSpec] = None):
    """Transposes `x` from the 'pipeline-major' layout to the 'layer-major' layout.

    Args:
        x: of shape [M + N - 1, N, ...], where x[t, i] represents the layerwise outputs of timestep[t] and layer[i].
        partition_spec: the partition spec for x.

    Returns:
        x': of shape [N, M, ...], where x'[i, j] represents layerwise outputs of pipeline layer[i]
            and microbatch[j]: x'[i, j] == x[i + j, i].
    """
    T, N = x.shape[:2]
    M = T - N + 1
    # [N, M+N-1, ...].
    x = jnp.transpose(x, [1, 0] + list(range(2, x.ndim)))
    # [N * (M+N-1), ...].
    x = jnp.reshape(x, [-1] + list(x.shape[2:]))
    # [N * (M+N), ...].
    x = jnp.pad(x, [(0, N)] + [(0, 0)] * (x.ndim - 1))
    # [N, M+N, ...].
    x = jnp.reshape(x, [N, M + N] + list(x.shape[1:]))
    # Apply sharding constraints at the first opportunity after reshapes
    # (i.e. when the input is first in the right shape for the constraint again).
    x = with_sharding_constraint(x, partition_spec)
    # [N, M, ...].
    x = x[:, :M]
    return x


class Pipeline(BaseLayer):
    """https://arxiv.org/abs/1811.06965."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("layer", None, "The param for the sub layer.")
        cfg.define("num_layers", None, "Repeat layers specified in `layer` this many times.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("layer", cfg.layer)

    def create_partition_specs_recursively(self) -> NestedParameterPartitionSpec:
        cfg = self.config
        specs = VDict(**super().create_partition_specs_recursively())

        def transform_factorization_spec(
            spec: Optional[FactorizationSpec],
        ) -> Optional[FactorizationSpec]:
            if spec is None:
                return None
            return FactorizationSpec(axes=[None] + list(spec.axes))

        return jax.tree_map(
            lambda spec: ParameterPartitionSpec(
                shape=(cfg.num_layers, *spec.shape),
                partition=PartitionSpec("pipeline", *spec.partition),
                factorization=transform_factorization_spec(spec.factorization),
            ),
            specs,
        )

    def initialize_parameters_recursively(
        self,
        prng_key: jax.random.KeyArray,
    ) -> NestedTensor:
        def init(prng_key_i):
            return VDict(layer=self.layer.initialize_parameters_recursively(prng_key_i))

        return jax.vmap(init)(self._split_keys(prng_key))

    class Output(NamedTuple):
        carry: NestedTensor
        ys: NestedTensor

    def _run(
        self,
        fn,
        carry=None,
        *,
        xs=None,
        carry_partition_spec: Optional[NestedPartitionSpec] = None,
        xs_partition_spec: Optional[NestedPartitionSpec] = None,
        ys_partition_spec: Optional[NestedPartitionSpec] = None,
    ):
        """Invokes 'fn' for each sub-layer with inputs already with the microbatch axis.

        Args:
            fn: A function with args (carry, x) returning a dict(carry=..., y=...).
            carry: a nested tensor for the iterative input of the 0'th sub-layer.
                It must have shape [M, microbatch_size, ...]
            xs: a nested tensor with separate inputs for each sub-layer,
                where each leaf value T is a tensor of shape [cfg.num_layers, M, microbatch_size, ...]
                and T[i, j, ...] represents layer-wise inputs of microbatch j to the i'th sub-layer.
            carry_partition_spec: partition spec for the carry tensors. If None, tensors will be replicated.
            xs_partition_spec: partition spec for the input xs tensors. If None, tensors will be replicated except for
                sharding along the "pipeline" mesh axis.
            ys_partition_spec: partition spec for the output ys tensors. If None, tensors will be replicated except for
                sharding along the "pipeline" mesh axis.

        Returns:
            A dict with the following keys:
            - carry: a nested tensor with the same structure as iterative_input_0
                representing the iterative output of the last sub-layer.
            - ys: a nested tensor where each leaf value T is a tensor of shape
                [cfg.num_layers, M, microbatch_size, ...] and T[i, ...] represents layer-wise output from
                the i'th sub-layer.
        """
        cfg = self.config
        self.vlog(1, "carry=%s xs=%s", shapes(carry), shapes(xs))
        # Number of microbatches.
        M = jax.tree_flatten(carry)[0][0].shape[0]
        # Number of pipeline stages.
        N = cfg.num_layers

        if carry is None:
            carry = dict()
            carry_partition_spec = dict()
        if carry_partition_spec is None:
            carry_partition_spec = jax.tree_map(
                lambda x: PartitionSpec(*[None for _ in x.shape]), carry
            )
        if xs is None:
            xs = dict()
            xs_partition_spec = dict()
        if xs_partition_spec is None:
            xs_partition_spec = jax.tree_map(
                lambda x: PartitionSpec("pipeline", *[None for _ in x.shape[1:]]), xs
            )

        def pad_carry(v_carry: Tensor, partition_spec: PartitionSpec):
            """Given input v_carry of shape [M, microbatch_size, ...], pads to [M + N - 1, N, microbatch_size, ...]."""
            # [M, 1, ...]
            v_carry = jnp.expand_dims(v_carry, 1)
            # Pad to shape [M + N - 1, N, ...].
            v_carry = jnp.pad(v_carry, [[0, N - 1], [0, N - 1]] + [(0, 0)] * (v_carry.ndim - 2))
            partition_spec = PartitionSpec(partition_spec[0], "pipeline", *partition_spec[1:])
            v_carry = with_sharding_constraint(v_carry, partition_spec)
            return v_carry

        padded_carry = jax.tree_map(pad_carry, carry, carry_partition_spec)
        # Transpose from "layer-major" [N, M, ...] to "pipeline-major" [N + M - 1, N, ...].
        #
        # Note: for efficient decoding we may want to skip transposes and keep decoding states in the "pipeline-major"
        # form (i.e., in the shape of [N + M - 1, N, ...]). To be investigated in the future.
        padded_xs = jax.tree_map(transpose_to_pipeline_stage_inputs, xs, xs_partition_spec)
        self.vlog(2, "padded_xs=%s", shapes(padded_xs))

        context = current_context()
        assert context is not None
        prng_keys = jax.random.split(context.prng_key, (M + N - 1) * N)

        def stack_and_reshape(*keys):
            keys = jnp.stack(keys)
            return jnp.reshape(keys, [M + N - 1, N] + list(keys.shape[1:]))

        prng_keys = jax.tree_map(stack_and_reshape, *prng_keys)
        with child_context("layer") as layer_context:

            def vmap_fn(state_n, prng_key_tn, carry_tn, x_tn):
                """Invokes fn for one microbatch and one layer.

                Args:
                    state_n: the parameters of the n'th layer.
                    prng_key_tn: the PRNG key for the v_carry'th timestep and n'th layer.
                    carry_tn: the carry input for the v_carry'th timestep and n'th layer.
                    x_tn: the xs input for the v_carry'th timestep and n'th layer.

                Returns:
                    dict(carry=<carry output>, y=<layerwise output>, output_collection=<auxiliary outputs>).
                """
                output_collection_tn = new_output_collection()
                with set_current_context(
                    layer_context.clone(
                        state=state_n,
                        prng_key=prng_key_tn,
                        output_collection=output_collection_tn,
                    )
                ):
                    carry_tn, y_tn = fn(carry_tn, x_tn)
                return dict(carry=carry_tn, y=y_tn, output_collection=output_collection_tn)

            def scan_fn(
                carry_output_t_1: NestedTensor,
                scan_t: Tuple[NestedTensor, NestedTensor, NestedTensor],
            ):
                """Processes timestep v_carry in the pipeline (in parallel across pipeline stages).

                Args:
                    carry_output_t_1: A NestedTensor where each Tensor has shape [N=num_layers, ...],
                        representing carry output of timestep {t-1}.
                    scan_t: A tuple of (prng_key_t, input_t, x_t), each is a NestedTensor where each leaf tensor has
                        shape [N, ...].

                Returns:
                    carry_output_t, dict(carry=..., y=..., output_collection=...), where
                    - `carry_output_t` and `carry` represents the carry output of timestep t and has the same structure
                       and shape as `carry_carry_output_t_1`;
                    - `y` is a NestedTensor representing the layerwise output of fn with leaves of shape [N, ...];
                    - `output_collection` is an OutputCollection representing the auxiliary outputs of fn with leaves
                       of shape [N, ...];
                """

                def compute_carry_input(v_input_t, v_carry_output_t_1):
                    """Computes the carry input for timestep v_carry.

                    Args:
                        v_input_t: a Tensor of shape [N, ...], where
                            v_input_t[0] of timestep t == microbatch[t] if t < M;
                            v_input_t[1:] are not used and are only there to make jnp.where() work shape-wise.
                        v_carry_output_t_1: a Tensor of shape [N, ...], representing carry output of timestep {t-1}.
                    """
                    # Move carry outputs from the previous timestep along the pipeline to the next stage so that the
                    # output from layer[j] becomes the input for layer[j+1].
                    v_carry_output_t_1 = jnp.roll(v_carry_output_t_1, 1, axis=0)
                    # The input to layer[0] comes from v_input_t[0].
                    return jnp.where(
                        jax.lax.broadcasted_iota("int32", v_input_t.shape, 0) == 0,
                        v_input_t,
                        v_carry_output_t_1,
                    )

                # Per-timestep inputs. Each leaf tensor has shape [N, ...].
                prng_key_t, input_t, x_t = scan_t
                carry_input_t = jax.tree_map(compute_carry_input, input_t, carry_output_t_1)

                # Parallel processing along the N axis.
                vmap_out = jax.vmap(vmap_fn)(layer_context.state, prng_key_t, carry_input_t, x_t)
                return vmap_out["carry"], vmap_out

            carry_t0 = jax.tree_map(
                lambda x: jnp.tile(jnp.zeros_like(x[:1]), [N] + [1] * (x.ndim - 1)), carry
            )
            self.vlog(
                2,
                "carry_t0=%s prng_keys=%s padded_carry=%s padded_xs=%s",
                shapes(carry_t0),
                shapes(prng_keys),
                shapes(padded_carry),
                shapes(padded_xs),
            )
            _, scan_ys = jax.lax.scan(
                scan_fn,
                init=carry_t0,
                xs=(prng_keys, padded_carry, padded_xs),
            )
            final_carry = jax.tree_map(lambda x: x[N - 1 :, -1, ...], scan_ys.pop("carry"))
            final_carry = jax.tree_map(with_sharding_constraint, final_carry, carry_partition_spec)

            ys = scan_ys["y"]
            if ys_partition_spec is None:
                ys_partition_spec = jax.tree_map(
                    lambda x: PartitionSpec("pipeline", *[None for _ in x.shape[1:]]), ys
                )
            # Transpose from pipeline-major [N + M - 1, N, ...] back to layer-major [N, M, ...].
            ys = jax.tree_map(transpose_from_pipeline_stage_outputs, ys, ys_partition_spec)
            scan_output_collection = jax.tree_map(
                transpose_from_pipeline_stage_outputs, scan_ys["output_collection"]
            )
            output_collection = layer_context.output_collection
            output_collection.summaries.update(**scan_output_collection.summaries)
            output_collection.state_updates.update(**scan_output_collection.state_updates)

        return self.Output(carry=final_carry, ys=ys)

    def _split_keys(self, prng_key: jax.random.KeyArray) -> jax.random.KeyArray:
        """Splits prng_key to num_layers keys iteratively and return the stacked keys.

        Args:
            prng_key: The input key.

        Returns:
            A stack of keys along axis 0. The result key array has shape [num_layers, ...] and can be used by
            jax.lax.scan().
        """
        cfg = self.config
        layer_prng_keys = []
        for _ in range(cfg.num_layers):
            # Generate the child keys iteratively to be consistent with how a parent module generates child module keys.
            prng_key, child_key = jax.random.split(prng_key)
            layer_prng_keys.append(child_key)
        return jax.tree_map(lambda *xs: jnp.stack(xs, axis=0), *layer_prng_keys)

    def _to_microbatches(self, inputs, *, microbatch_size=-1, num_microbatches=-1):
        if microbatch_size < 0 and num_microbatches < 0:
            raise ValueError(
                "At least one of microbatch_size and num_microbatches must be specified."
            )
        return jax.tree_map(
            lambda x: jnp.reshape(x, [num_microbatches, microbatch_size] + list(x.shape[1:])),
            inputs,
        )

    def _from_microbatches(self, inputs):
        return jax.tree_map(lambda x: jnp.reshape(x, [-1] + list(x.shape[2:])), inputs)

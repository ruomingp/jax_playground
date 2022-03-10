"""A generic repeat layer.

Adapted from:
https://github.com/tensorflow/lingvo/blob/master/lingvo/core/repeat_layer.py
https://github.com/tensorflow/lingvo/blob/master/lingvo/jax/layers/repeats.py

A repeat layer consists a stack of N identical sub layers, where
  * The variables are stacked across layers. Each stacked variable has shape [N, ...].
  * The computation is performed with a recurrent loop across layers.

Compared with a layer stack, a repeat layer's XLA code size does not grow
proportional to the number of layers. It also reduces HBM usage but incurs
additional computation through rematerialization.

Repeat._run() allows its subclasses to describe arbitrary
computation across sub layers.

Inputs to repeat layer computation fall into two categories:

  * carry: iterative input to the first sub layer, e.g., hidden vectors.
  * xs: separate inputs for each sub layer, specified by tensors of shape [N, ...],
      where T[i, ...] is the input for sub layer i, e.g., states for auto-regressive inference.

The output of a sub layer can include:

  * carry: iterative input for the next sub layer.
  * ys: layer-wise outputs, to be stacked for the final output, e.g.,
      updated_states from auto-regressive inference.

The final output of a repeat layer will include:

  * carry: the iterative output of the final sub layer.
  * ys: stacked tensors of layer-wise outputs, of shape [N, ...],
      where T[i, ...] is a layer-wise output of sub layer i.

In pseudo code::

  def _run(theta, fn, carry, xs):
      for i in range(p.num_layers):
          carry, ys[i, ...] = fn(carry, xs[i, ...])
      return carry, ys

TODO(rpang): reduce memory usage with custom checkpoint policy, as in:
https://github.com/tensorflow/lingvo/blob/883352f795366b5489949b893c0cf1165e99ea17/lingvo/jax/layers/recurrent.py#L415-L446
"""

from typing import NamedTuple, Optional

import jax
from jax import numpy as jnp

import config as config_lib
from module import (
    BaseLayer,
    FactorizationSpec,
    Module,
    NestedParameterPartitionSpec,
    NestedPartitionSpec,
    NestedTensor,
    ParameterPartitionSpec,
    PartitionSpec,
    child_context,
    current_context,
    new_output_collection,
    set_current_context,
)
from utils import VDict


class Repeat(BaseLayer):
    """A layer which repeats a sub layer sequentially using a jax.lax.scan loop."""

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
                partition=PartitionSpec(None, *spec.partition),
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

    def _run(self, fn, carry=None, *, xs=None):
        """Invokes 'fn' for each sub-layer.

        Args:
            fn: A function with args (carry, x) returning a dict(carry=..., y=...).
            carry: a nested tensor for the iterative input of the 0'th sub-layer.
            xs: a nested tensor with separate inputs for each sub-layer,
                where each leaf value T is a tensor of shape [cfg.num_layers, ...]
                and T[i, ...] represents layer-wise inputs to the i'th sub-layer.

        Returns:
            A dict with the following keys:
            - carry: a nested tensor with the same structure as iterative_input_0
                representing the iterative output of the last sub-layer.
            - ys: a nested tensor where each leaf value T is a tensor of shape [cfg.num_layers, ...] and
                T[i, ...] represents layer-wise output from the i'th sub-layer.
        """
        if xs is None:
            xs = dict()
        if carry is None:
            carry = dict()

        context = current_context()
        assert context is not None
        prng_key = context.prng_key
        with child_context("layer") as layer_context:

            def scan_fn(carry_i, scan_i):
                prng_key_i, layer_state_i, x_i = scan_i
                output_collection_i = new_output_collection()
                with set_current_context(
                    layer_context.clone(
                        state=layer_state_i,
                        prng_key=prng_key_i,
                        output_collection=output_collection_i,
                    )
                ):
                    carry_i, y_i = fn(carry_i, x_i)
                return carry_i, dict(y_i=y_i, output_collection=output_collection_i)

            carry, scan_ys = jax.lax.scan(
                scan_fn,
                init=carry,
                xs=(self._split_keys(prng_key), layer_context.state, xs),
            )

            output_collection = layer_context.output_collection
            output_collection.summaries.update(**scan_ys["output_collection"].summaries)
            output_collection.state_updates.update(**scan_ys["output_collection"].state_updates)

        return self.Output(carry=carry, ys=scan_ys["y_i"])

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

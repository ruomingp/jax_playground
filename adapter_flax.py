"""Adapter layers to use Flax/Linen modules.

FlaxLayer allows users to use flax.linen modules in an ajax module hierarchy.
See the FeedForward layer in adapter_flax_test.py for an example.
"""
import collections
from typing import Callable

import jax.random
from flax.linen import Module as FlaxModule

import config as config_lib
import utils
from module import BaseLayer, Module, NestedParameterPartitionSpec, NestedTensor


class FlaxLayer(BaseLayer):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("create_module_fn", None, "A function to return a linen.Module.")
        cfg.define("create_module_kwargs", {}, "The kwargs for create_module_fn.")
        cfg.define(
            "create_dummy_input_fn",
            None,
            "A function to return (args, kwargs) used for linen.Module.init.",
        )
        cfg.define("create_dummy_input_kwargs", {}, "The kwargs for create_dummy_input_fn.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        self._module = self._create_flax_module()
        self.vlog(1, "module=%s", self._module)
        self._dummy_inputs = self._create_dummy_inputs()
        self.vlog(1, "dummy_inputs=%s", utils.shapes(self._dummy_inputs))

    def create_partition_specs_recursively(self) -> NestedParameterPartitionSpec:
        return self.config.param_partition_spec

    def _create_flax_module(self) -> FlaxModule:
        cfg = self.config
        return cfg.create_module_fn(**cfg.create_module_kwargs)

    def _create_dummy_inputs(self):
        cfg = self.config
        return cfg.create_dummy_input_fn(**cfg.create_dummy_input_kwargs)

    def initialize_parameters_recursively(self, prng_key: jax.random.KeyArray) -> NestedTensor:
        args, kwargs = self._dummy_inputs
        return self._module.init(prng_key, *args, **kwargs)

    def forward(self, *args, mutable=None, **kwargs):
        if mutable is None:
            mutable = "batch_stats" if self.is_training else False
        apply_outputs = self._module.apply(
            self.parameters,
            *args,
            mutable=mutable,
            rngs=collections.defaultdict(lambda: self.prng_key),
            **kwargs,
        )
        if mutable:
            outputs, variable_updates = apply_outputs
            self.vlog(3, "variable_updates=%s", variable_updates)
            for name, value in variable_updates.items():
                self.add_state_update(name, value)
        else:
            outputs = apply_outputs
        return outputs


def config_for_flax_module(
    create_module_fn: Callable[[], FlaxModule],
    create_dummy_input_fn: Callable[[], NestedTensor],
    **kwargs,
):
    return FlaxLayer.default_config().set(
        create_module_fn=create_module_fn, create_dummy_input_fn=create_dummy_input_fn, **kwargs
    )

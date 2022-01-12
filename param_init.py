import math
from typing import Sequence, Tuple

import jax.random
from jax import numpy as jnp

from typing import Any

import config as config_lib

Shape = Sequence[int]


class Initializer:
    def initialize(
        self,
        name: str,
        *,
        prng_key: jax.random.KeyArray,
        shape: Shape,
        dtype: jnp.dtype,
    ) -> jnp.ndarray:
        raise NotImplementedError(type(self))


class ConstantInitializer(Initializer):

    def __init__(self, value: Any):
        self._value = value

    def initialize(
        self,
        name: str,
        *,
        prng_key: jax.random.KeyArray,
        shape: Shape,
        dtype: jnp.dtype,
    ) -> jnp.ndarray:
        return jnp.full(shape=shape, fill_value=self._value, dtype=dtype)


class DefaultInitializer(config_lib.Configurable, Initializer):
    """The default initializer."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("gain", 1.0, "The default gain for weight initialization.")
        cfg.define(
            "fan",
            "xavier",
            'How to compute the fan, supported values are "fan_in", "fan_out", and "xavier".',
        )
        cfg.define(
            "distribution", "uniform", 'Weight distribution: "uniform" or "normal".'
        )
        return cfg

    def initialize(
        self,
        name: str,
        *,
        prng_key: jax.random.KeyArray,
        shape: Shape,
        dtype: jnp.dtype,
    ) -> jnp.ndarray:
        if name.endswith("bias"):
            return jnp.zeros(shape, dtype=dtype)
        elif name.endswith("scale"):
            return jnp.ones(shape, dtype=dtype)
        elif name.endswith("weight"):
            return self._initialize_weight(
                name, prng_key=prng_key, shape=shape, dtype=dtype
            )
        else:
            raise NotImplementedError(f"Unsupported parameter name ({name})")

    def _initialize_weight(
        self,
        name: str,
        *,
        prng_key: jax.random.KeyArray,
        shape: Shape,
        dtype: jnp.dtype,
    ) -> jnp.ndarray:
        cfg = self.config
        fan = self._get_fan(name, shape)
        std = cfg.gain / math.sqrt(fan)
        if cfg.distribution == "uniform":
            b = math.sqrt(3) * std
            weight = jax.random.uniform(
                prng_key, shape=shape, dtype=dtype, minval=-b, maxval=b
            )
        elif cfg.distribution == "normal":
            weight = jax.random.normal(prng_key, shape=shape, dtype=dtype) * std
        else:
            raise NotImplementedError(f"Unsupported distribution ({cfg.distribution})")
        return weight

    def _get_fan(self, name: str, shape: Shape):
        cfg = self.config
        fan_in, fan_out = self._calculate_fan_in_and_fan_out(name, shape)
        if cfg.fan == "fan_in":
            return fan_in
        elif cfg.fan == "fan_out":
            return fan_out
        elif cfg.fan == "xavier":
            return (fan_in + fan_out) / 2
        else:
            raise NotImplementedError(f"Unsupported fan ({cfg.fan})")

    def _calculate_fan_in_and_fan_out(self, name: str, shape: Shape) -> Tuple[int, int]:
        if len(shape) < 2:
            raise NotImplementedError(f"Unsupported weight shape {shape} for {name}")
        output_dim = shape[-1]
        input_dim = shape[-2]
        receptive_field_size = 1
        if len(shape) > 2:
            for s in shape[:-2]:
                receptive_field_size *= s
        fan_in = input_dim * receptive_field_size
        fan_out = output_dim * receptive_field_size
        return fan_in, fan_out

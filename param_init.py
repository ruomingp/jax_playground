"""Modules for configurable parameter initialization."""

from typing import Any, NamedTuple, Sequence, Tuple, Union

import jax
from jax import numpy as jnp

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


class GaussianInitializer(Initializer):
    def __init__(self, std: float):
        self._std = std

    def initialize(
        self,
        name: str,
        *,
        prng_key: jax.random.KeyArray,
        shape: Shape,
        dtype: jnp.dtype,
    ) -> jnp.ndarray:
        return jax.random.normal(prng_key, shape=shape, dtype=dtype) * self._std


def truncated_normal(stddev: float = 1e-2, dtype: jnp.dtype = jnp.float_):
    """Truncated normal variant of jax.nn.initializers.

    Args:
        stddev: Standard deviation of gaussian to draw from.
        dtype: Type to draw from.
    Returns:
        initializer fn.
    """

    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        # constant is stddev of standard normal truncated to (-2, 2)
        stddev = stddev / jnp.array(0.87962566103423978, dtype)
        return jax.random.truncated_normal(key, -2, 2, shape, dtype) * stddev

    return init


class FanAxes(NamedTuple):
    # Input axis or sequence of axes of the fan "input" dimension.
    in_axis: Union[Tuple[int], int]
    # Output axis or sequence of axes of the fan "output" dimension.
    out_axis: Union[Tuple[int], int]


class DefaultInitializer(config_lib.Configurable, Initializer):
    """The default initializer."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("scale", 1.0, "The default scale for weight initialization.")
        cfg.define(
            "fan",
            "fan_avg",
            (
                'Type of fan to compute, supported values are "fan_in", "fan_out", "fan_avg" and None. '
                + "If None then no fan scaling factor is computed."
            ),
        )
        cfg.define(
            "distribution",
            "uniform",
            'Weight distribution: "uniform", "normal", or "truncated_normal".',
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
            return self._initialize_weight(name, prng_key=prng_key, shape=shape, dtype=dtype)
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
        if cfg.fan is not None:
            fan_axes = self._compute_fan_axes(name, shape)
            initializer = jax.nn.initializers.variance_scaling(
                cfg.scale,
                mode=cfg.fan,
                distribution=cfg.distribution,
                dtype=dtype,
                **fan_axes._asdict(),
            )
            return initializer(prng_key, shape=shape)
        elif cfg.distribution == "uniform":
            initializer = jax.nn.initializers.uniform(cfg.scale, dtype=dtype)
        elif cfg.distribution == "normal":
            initializer = jax.nn.initializers.normal(cfg.scale, dtype=dtype)
        elif cfg.distribution == "truncated_normal":
            initializer = truncated_normal(cfg.scale, dtype=dtype)
        else:
            raise NotImplementedError(
                f"Unsupported fan {cfg.fan} and distribution {cfg.distribution}."
            )
        return initializer(prng_key, shape=shape)

    def _compute_fan_axes(self, name: str, shape: Shape) -> FanAxes:
        # Override for custom fan behavior.
        return FanAxes(in_axis=-2, out_axis=-1)

from typing import Callable

import jax
from jax import nn
from jax import numpy as jnp

import module
from module import Module, NestedParameters

Tensor = jnp.ndarray

enable_numeric_checks = False


def get_activation_fn(name) -> Callable[[Tensor], Tensor]:
    if name.startswith("nn."):
        return getattr(nn, name[3:])
    else:
        raise NotImplementedError(f"Unsupported activation function {name}")


def check_numerics(x: Tensor, msg_fmt: str = "", **msg_kwargs):
    global enable_numeric_checks
    if enable_numeric_checks:
        assert bool(
            jnp.isfinite(x).all()
        ), f"Check numerics {msg_fmt.format(**msg_kwargs)}: {x}"
    return x


class Dropout(Module):
    """The dropout layer."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("rate", 0, "The dropout rate (i.e., 1 - keep_prob).")
        return cfg

    def _initialize_module_parameters(
        self, *, prng_key: jax.random.KeyArray
    ) -> NestedParameters:
        return {}

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        if not self.is_training or cfg.rate == 0:
            return x
        assert 0 < cfg.rate < 1
        dropout = jax.random.bernoulli(
            module.current_context().prng_key, p=cfg.rate, shape=x.shape
        )
        return jnp.where(dropout, jnp.zeros_like(x), x / (1.0 - cfg.rate))


class LayerNorm(Module):
    """Reference:"""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("dim", 0, "Input feature dim.")
        cfg.define("eps", 1e-8, "The epsilon.")
        return cfg

    def _initialize_module_parameters(
        self, *, prng_key: jax.random.KeyArray
    ) -> NestedParameters:
        cfg = self.config
        return {
            "scale": jnp.ones(shape=[cfg.dim], dtype=self.dtype()),
            "bias": jnp.zeros(shape=[cfg.dim], dtype=self.dtype()),
        }

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        x_dtype = x.dtype
        x = x.astype(jnp.float32)
        x -= x.mean(axis=-1, keepdims=True)
        variance = (x * x).mean(axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + cfg.eps)
        x = x.astype(x_dtype)
        x = x * self.parameters["scale"] + self.parameters["bias"]
        return x


class RMSNorm(Module):
    """Reference: https://github.com/bzhangGo/rmsnorm."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("dim", 0, "Input feature dim.")
        cfg.define("eps", 1e-8, "The epsilon.")
        return cfg

    def _initialize_module_parameters(
        self, *, prng_key: jax.random.KeyArray
    ) -> NestedParameters:
        cfg = self.config
        return {"scale": jnp.ones(shape=[cfg.dim], dtype=self.dtype())}

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        x_dtype = x.dtype
        x = x.astype(jnp.float32)
        moment2 = (x * x).mean(axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(moment2 + cfg.eps)
        x = x.astype(x_dtype)
        x = x * self.parameters["scale"]
        return x


class BatchNorm(Module):
    """Reference:"""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("dim", 0, "Input feature dim.")
        cfg.define("decay", 0.999, "The decay for computing moving mean/variance.")
        cfg.define("eps", 1e-8, "The epsilon.")
        return cfg

    def _initialize_module_parameters(
        self, *, prng_key: jax.random.KeyArray
    ) -> NestedParameters:
        cfg = self.config
        return {
            "scale": jnp.ones(shape=[cfg.dim], dtype=self.dtype()),
            "bias": jnp.zeros(shape=[cfg.dim], dtype=self.dtype()),
            "moving_mean": jnp.zeros(shape=[cfg.dim], dtype=jnp.float32),
            "moving_variance": jnp.ones(shape=[cfg.dim], dtype=jnp.float32),
        }

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        x_dtype = x.dtype
        x = x.astype(jnp.float32)
        reduction_axis = list(range(x.ndim - 1))
        if self.is_training:
            mean = jnp.mean(x, axis=reduction_axis, keepdims=True)
            variance = jnp.mean((x - mean) ** 2, axis=reduction_axis, keepdims=True)
            self.add_parameter_update(
                "moving_mean",
                cfg.decay * self.parameters["moving_mean"] + (1 - cfg.decay) * mean,
            )
            self.add_parameter_update(
                "moving_variance",
                cfg.decay * self.parameters["moving_variance"]
                + (1 - cfg.decay) * variance,
            )
        else:
            mean = self.parameters["moving_mean"]
            variance = self.parameters["moving_variance"]
        x = (x - mean) * jax.lax.rsqrt(variance + cfg.eps)
        x = x.astype(x_dtype)
        x = x * self.parameters["scale"] + self.parameters["bias"]
        return x


class Linear(Module):
    """The linear layer."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("input_dim", 0, "Input feature dim.")
        cfg.define("output_dim", 0, "Output feature dim.")
        cfg.define("bias", True, "Whether to add a bias.")
        return cfg

    def _initialize_module_parameters(
        self, *, prng_key: jax.random.KeyArray
    ) -> NestedParameters:
        cfg = self.config
        params = dict(
            weight=self._initialize_parameter(
                "weight", prng_key=prng_key, shape=(cfg.input_dim, cfg.output_dim)
            )
        )
        if cfg.bias:
            params["bias"] = jnp.zeros(shape=[cfg.output_dim], dtype=self.dtype())
        return params

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.parameters["weight"] + self.parameters.get("bias", 0)


class Conv2D(Module):
    """The 2-D convolution layer.

    Kernel weights have the HWIO layout and in the shape of (window[0], window[1], input_dim, output_dim).
    Both inputs and outputs will be in the NHWC layout.
    """

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("window", (1, 1), "The convolution window.")
        cfg.define("strides", (1, 1), "The convolution strides.")
        cfg.define(
            "padding", ((0, 0), (0, 0)), "Paddings ((top, bottom), (left, right))."
        )
        cfg.define("input_dim", 0, "Input feature dim.")
        cfg.define("output_dim", 0, "Output feature dim.")
        cfg.define("bias", True, "Whether to add a bias.")
        return cfg

    def _initialize_module_parameters(
        self, *, prng_key: jax.random.KeyArray
    ) -> NestedParameters:
        cfg = self.config
        if cfg.padding in ("SAME", "VALID"):
            if cfg.padding == "SAME" and any(s > 1 for s in cfg.strides):
                raise NotImplementedError("SAME padding does not support strides > 1")
        else:
            ((top, bottom), (left, right)) = cfg.padding
            if any(p < 0 for p in (top, bottom, left, right)):
                raise NotImplementedError("Negative padding is not supported")
        params = dict(
            weight=self._initialize_parameter(
                "weight",
                prng_key=prng_key,
                shape=list(cfg.window) + [cfg.input_dim, cfg.output_dim],
            )
        )
        if cfg.bias:
            params["bias"] = jnp.zeros(shape=[cfg.output_dim], dtype=self.dtype())
        return params

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        return jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.parameters["weight"],
            window_strides=cfg.strides,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            padding=cfg.padding,
        ) + self.parameters.get("bias", 0)

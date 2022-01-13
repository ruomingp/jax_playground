from typing import Callable, Dict

import jax
from jax import nn
from jax import numpy as jnp

from module import BaseLayer, NestedTensor, ParameterSpec, PartitionSpec
import param_init

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


class Dropout(BaseLayer):
    """The dropout layer."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("rate", 0, "The dropout rate (i.e., 1 - keep_prob).")
        return cfg

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        if not self.is_training or cfg.rate == 0:
            return x
        assert 0 < cfg.rate < 1
        dropout = jax.random.bernoulli(
            BaseLayer.current_context().prng_key, p=cfg.rate, shape=x.shape
        )
        return jnp.where(dropout, jnp.zeros_like(x), x / (1.0 - cfg.rate))


class LayerNorm(BaseLayer):
    """Reference:"""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("dim", 0, "Input feature dim.")
        cfg.define("eps", 1e-8, "The epsilon.")
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        return {
            "scale": ParameterSpec(
                shape=[cfg.dim], partition_spec=PartitionSpec("model")
            ),
            "bias": ParameterSpec(
                shape=[cfg.dim], partition_spec=PartitionSpec("model")
            ),
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


class RMSNorm(BaseLayer):
    """Reference: https://github.com/bzhangGo/rmsnorm."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("dim", 0, "Input feature dim.")
        cfg.define("eps", 1e-8, "The epsilon.")
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        return {
            "scale": ParameterSpec(
                shape=[cfg.dim], partition_spec=PartitionSpec("model")
            ),
        }

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        x_dtype = x.dtype
        x = x.astype(jnp.float32)
        moment2 = (x * x).mean(axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(moment2 + cfg.eps)
        x = x.astype(x_dtype)
        x = x * self.parameters["scale"]
        return x


class BatchNorm(BaseLayer):
    """Reference:"""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("dim", 0, "Input feature dim.")
        cfg.define("decay", 0.999, "The decay for computing moving mean/variance.")
        cfg.define("eps", 1e-8, "The epsilon.")
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        return {
            "scale": ParameterSpec(
                shape=[cfg.dim], partition_spec=PartitionSpec("model")
            ),
            "bias": ParameterSpec(
                shape=[cfg.dim], partition_spec=PartitionSpec("model")
            ),
            "moving_mean": ParameterSpec(
                shape=[cfg.dim],
                dtype=jnp.float32,
                partition_spec=PartitionSpec("model"),
                initializer=param_init.ConstantInitializer(0.0),
            ),
            "moving_variance": ParameterSpec(
                shape=[cfg.dim],
                dtype=jnp.float32,
                partition_spec=PartitionSpec("model"),
                initializer=param_init.ConstantInitializer(1.0),
            ),
        }

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        x_dtype = x.dtype
        x = x.astype(jnp.float32)
        reduction_axis = tuple(range(x.ndim - 1))
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


class Linear(BaseLayer):
    """The linear layer."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("input_dim", 0, "Input feature dim.")
        cfg.define("output_dim", 0, "Output feature dim.")
        cfg.define("bias", True, "Whether to add a bias.")
        cfg.param_partition_spec = PartitionSpec(None, None)
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        params = dict(
            weight=ParameterSpec(
                shape=(cfg.input_dim, cfg.output_dim),
                partition_spec=cfg.param_partition_spec))
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim],
                partition_spec=PartitionSpec(cfg.param_partition_spec[-1]))
        return params

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.parameters["weight"] + self.parameters.get("bias", 0)


class Conv2D(BaseLayer):
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
        cfg.param_partition_spec = PartitionSpec(None, None, None, None)
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        if cfg.padding in ("SAME", "VALID"):
            if cfg.padding == "SAME" and any(s > 1 for s in cfg.strides):
                raise NotImplementedError("SAME padding does not support strides > 1")
        else:
            ((top, bottom), (left, right)) = cfg.padding
            if any(p < 0 for p in (top, bottom, left, right)):
                raise NotImplementedError("Negative padding is not supported")
        params = dict(
            weight=ParameterSpec(
                shape=list(cfg.window) + [cfg.input_dim, cfg.output_dim],
                partition_spec=cfg.param_partition_spec))
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim],
                partition_spec=PartitionSpec(cfg.param_partition_spec[-1]))
        return params

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        return (
            jax.lax.conv_general_dilated(
                lhs=x,
                rhs=self.parameters["weight"],
                window_strides=cfg.strides,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
                padding=cfg.padding,
            )
            + self.parameters.get("bias", 0)
        )

"""Basic layers."""

import math
from typing import Callable, Dict

import jax
import numpy as np
from jax import nn
from jax import numpy as jnp

import config
import param_init
from module import BaseLayer, FactorizationSpec, ParameterSpec
from utils import Tensor


def get_activation_fn(name) -> Callable[[Tensor], Tensor]:
    if name.startswith("nn."):
        return getattr(nn, name[3:])
    else:
        raise NotImplementedError(f"Unsupported activation function {name}")


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
        samples = jax.random.uniform(
            self.prng_key, shape=x.shape, dtype=x.dtype, minval=0.0, maxval=1.0
        )
        dropout = jnp.floor(1 - cfg.rate + samples)
        return x * dropout / (1.0 - cfg.rate)


def set_dropout_rate_recursively(cfg: config.Config, dropout_rate: float):
    def is_dropout_config(cfg):
        return isinstance(cfg, config.InstantiableConfig) and issubclass(cfg.cls, Dropout)

    def visit_fn(key, value):
        if is_dropout_config(value):
            value.rate = dropout_rate

    def enter_fn(key, value, default_kv):
        return None if is_dropout_config(value) else default_kv

    cfg.visit(visit_fn=visit_fn, enter_fn=enter_fn)


class LayerNorm(BaseLayer):
    """Reference: https://arxiv.org/abs/1607.06450."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("dim", 0, "Input feature dim.")
        cfg.define("eps", 1e-8, "The epsilon.")
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        return {
            "scale": ParameterSpec(shape=[cfg.dim], partition=(None,)),
            "bias": ParameterSpec(shape=[cfg.dim], partition=(None,)),
        }

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        x_dtype = x.dtype
        x = x.astype(jnp.float32)
        x_mean = x.mean(axis=-1, keepdims=True)
        x -= x_mean
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
            "scale": ParameterSpec(shape=[cfg.dim], partition=(None,)),
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


class GroupNorm(BaseLayer):
    """https://arxiv.org/abs/1803.08494."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("dim", 0, "Input feature dim. Required.")
        cfg.define("num_groups", 0, "The number of groups.")
        cfg.define("eps", 1e-8, "The epsilon.")
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        return {
            "scale": ParameterSpec(shape=[cfg.dim], partition=(None,)),
            "bias": ParameterSpec(shape=[cfg.dim], partition=(None,)),
        }

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        if cfg.num_groups <= 0 or cfg.dim % cfg.num_groups != 0:
            raise ValueError(f"num_groups ({cfg.num_groups}) must divide dim ({cfg.dim})")
        group_size = cfg.dim // cfg.num_groups
        x_dtype = x.dtype
        x = x.astype(jnp.float32)
        # Reshape to [..., num_groups, group_size].
        y = jnp.reshape(x, x.shape[:-1] + [cfg.num_groups, group_size])
        # Reduce along spatial dims and group_size, but not along batch or num_groups.
        reduction_axis = list(range(1, y.ndim - 2)) + [-1]
        mean = jnp.mean(y, axis=reduction_axis, keepdims=True)
        variance = jnp.mean((y - mean) ** 2, axis=reduction_axis, keepdims=True)
        y = (y - mean) * jax.lax.rsqrt(variance + cfg.eps)
        x = y.astype(x_dtype)
        x = x * self.parameters["scale"] + self.parameters["bias"]
        return x


class BatchNorm(BaseLayer):
    """https://arxiv.org/abs/1502.03167."""

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
            "scale": ParameterSpec(shape=[cfg.dim], partition=(None,)),
            "bias": ParameterSpec(shape=[cfg.dim], partition=(None,)),
            "moving_mean": ParameterSpec(
                shape=[cfg.dim],
                dtype=jnp.float32,
                partition=(None,),
                initializer=param_init.ConstantInitializer(0.0),
            ),
            "moving_variance": ParameterSpec(
                shape=[cfg.dim],
                dtype=jnp.float32,
                partition=(None,),
                initializer=param_init.ConstantInitializer(1.0),
            ),
        }

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        x_dtype = x.dtype
        x = x.astype(jnp.float32)
        reduction_axis = tuple(range(x.ndim - 1))
        if self.is_training:
            mean = jnp.mean(x, axis=reduction_axis)
            variance = jnp.mean((x - mean) ** 2, axis=reduction_axis)
            self.add_state_update(
                "moving_mean",
                cfg.decay * self.parameters["moving_mean"] + (1 - cfg.decay) * mean,
            )
            self.add_state_update(
                "moving_variance",
                cfg.decay * self.parameters["moving_variance"] + (1 - cfg.decay) * variance,
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
        cfg.param_partition_spec = (None, None)
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        params = dict(
            weight=ParameterSpec(
                shape=(cfg.input_dim, cfg.output_dim),
                partition=cfg.param_partition_spec,
                factorization=FactorizationSpec(axes=("row", "col")),
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim], partition=(cfg.param_partition_spec[-1],)
            )
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
        cfg.define("padding", ((0, 0), (0, 0)), "Paddings ((top, bottom), (left, right)).")
        cfg.define("input_dim", 0, "Input feature dim.")
        cfg.define("output_dim", 0, "Output feature dim.")
        cfg.define("bias", True, "Whether to add a bias.")
        cfg.param_partition_spec = (None, None, None, None)
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
                partition=cfg.param_partition_spec,
                factorization=FactorizationSpec(axes=(None, None, "row", "col")),
            )
        )
        if cfg.bias:
            params["bias"] = ParameterSpec(
                shape=[cfg.output_dim], partition=(cfg.param_partition_spec[-1],)
            )
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


class Embedding(BaseLayer):
    """Implements an embedding lookup function.

    Batched map for int in [0, <num_embeddings>) -> <dim> float vector.
    """

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("num_embeddings", 0, "Maximum number of embeddings in table.")
        cfg.define("dim", 0, "Embedding vector dimensionality.")
        cfg.param_partition_spec = (None, "model")
        # By default, initialize to Gaussian with std=1/sqrt(dim), e.g., 0.036 when dim=768.
        #
        # This is the same as:
        # https://github.com/google-research/t5x/blob/f7978d63448c43bdb339ae73fa557ba472be30d6/t5x/examples/scalable_t5/layers.py#L535
        #
        # PyTorch uses normal with std=1.0, regardless of dim/size:
        # https://github.com/pytorch/pytorch/blob/febff45900e57d3e05ee72c1ecfe7d4fcbc582d9/torch/nn/modules/sparse.py#L149
        #
        # TensorFlow/Haiku use truncated normal with std=1.0
        # https://github.com/deepmind/dm-haiku/blob/220c6b02a22f1ee9bea7dc8e017f3090108f75e4/haiku/_src/embed.py#L117
        cfg.param_init = param_init.DefaultInitializer.default_config().set(
            fan="fan_out", distribution="normal"
        )
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        return dict(
            weight=ParameterSpec(
                shape=[cfg.num_embeddings, cfg.dim],
                partition=cfg.param_partition_spec,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        emb = self.parameters["weight"]
        return emb[x]

    def attend(self, x: Tensor) -> Tensor:
        """Apply query array 'x' to the embedding weight array.

        Args:
            x: array where last dimension equals 'dim'.
        Returns:
            Result of batched inner product of 'x' and embedding weight.
        """
        return jnp.einsum("bld,nd->bln", x, self.parameters["weight"])

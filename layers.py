from typing import Callable

import jax
from jax import nn
from jax import numpy as jnp

import module
from module import Module, NestedParameters

Tensor = jnp.ndarray

enable_numeric_checks = False


def get_activation_fn(name) -> Callable[[Tensor], Tensor]:
    if name.startswith('nn.'):
        return getattr(nn, name[3:])
    else:
        raise NotImplementedError(f'Unsupported activation function {name}')


def check_numerics(x: Tensor, msg_fmt: str = '', **msg_kwargs):
    global enable_numeric_checks
    if enable_numeric_checks:
        assert bool(jnp.isfinite(x).all()), f'Check numerics {msg_fmt.format(**msg_kwargs)}: {x}'
    return x


class Dropout(Module):
    """The dropout layer."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define('rate', 0, 'The dropout rate (i.e., 1 - keep_prob).')
        return cfg

    def _initialize_module_parameters(self, *, prng_key: jax.random.KeyArray) -> NestedParameters:
        return {}

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        if not self.is_training or cfg.rate == 0:
            return x
        assert 0 < cfg.rate < 1
        dropout = jax.random.bernoulli(module.current_context().prng_key, p=cfg.rate, shape=x.shape)
        return jnp.where(dropout, jnp.zeros_like(x), x / (1. - cfg.rate))


class Linear(Module):
    """The linear layer."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define('input_dim', 0, 'Input feature dim.')
        cfg.define('output_dim', 0, 'Output feature dim.')
        cfg.define('bias', True, 'Whether to add a bias.')
        return cfg

    def _initialize_module_parameters(self, *, prng_key: jax.random.KeyArray) -> NestedParameters:
        cfg = self.config
        params = dict(
            weight=self._initialize_parameter('weight', prng_key=prng_key, shape=(cfg.input_dim, cfg.output_dim)))
        if cfg.bias:
            params['bias'] = jnp.zeros(shape=[cfg.output_dim], dtype=self.dtype())
        return params

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.parameters['weight'] + self.parameters.get('bias', 0)


class LayerNorm(Module):
    """Reference: """

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define('dim', 0, 'Input feature dim.')
        cfg.define('eps', 1e-8, 'The epsilon.')
        return cfg

    def _initialize_module_parameters(self, *, prng_key: jax.random.KeyArray) -> NestedParameters:
        cfg = self.config
        return {'scale': jnp.ones(shape=[cfg.dim], dtype=self.dtype()),
                'bias': jnp.zeros(shape=[cfg.dim], dtype=self.dtype())}

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        x_dtype = x.dtype
        x = x.astype(jnp.float32)
        x -= x.mean(axis=-1, keepdims=True)
        variance = (x * x).mean(axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + cfg.eps)
        x = x.astype(x_dtype)
        x = x * self.parameters['scale'] + self.parameters['bias']
        return x


class RMSNorm(Module):
    """Reference: https://github.com/bzhangGo/rmsnorm."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define('dim', 0, 'Input feature dim.')
        cfg.define('eps', 1e-8, 'The epsilon.')
        return cfg

    def _initialize_module_parameters(self, *, prng_key: jax.random.KeyArray) -> NestedParameters:
        cfg = self.config
        return {'scale': jnp.ones(shape=[cfg.dim], dtype=self.dtype())}

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        x_dtype = x.dtype
        x = x.astype(jnp.float32)
        moment2 = (x * x).mean(axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(moment2 + cfg.eps)
        x = x.astype(x_dtype)
        x = x * self.parameters['scale']
        return x

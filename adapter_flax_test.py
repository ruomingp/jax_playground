import jax.random
from absl.testing import absltest
from flax import linen as nn
from flax.core import FrozenDict
from jax import numpy as jnp

import config as config_lib
import utils
from adapter_flax import config_for_flax_module
from module import BaseLayer, Module
from module import functional as F
from test_utils import TestCase


class FeedForward(BaseLayer):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("input_dim", 0, "The input feature dim.")
        cfg.define("hidden_dim", 0, "The hidden feature dim.")
        cfg.define("output_dim", 0, "The output feature dim.")
        cfg.define(
            "linear",
            config_for_flax_module(nn.Dense, cls.dummy_inputs),
            "The config for linear layers.",
        )
        cfg.define(
            "norm",
            config_for_flax_module(nn.BatchNorm, cls.dummy_inputs_for_norm),
            "The config for the norm layer.",
        )
        return cfg

    @classmethod
    def dummy_inputs(cls, dim, dtype):
        return (jnp.zeros([0, 0, dim], dtype=dtype),), {}

    @classmethod
    def dummy_inputs_for_norm(cls, dim, dtype):
        args, kwargs = cls.dummy_inputs(dim, dtype)
        kwargs["use_running_average"] = False
        return args, kwargs

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "linear1",
            cfg.linear.set(
                create_module_kwargs=dict(features=cfg.hidden_dim),
                create_dummy_input_kwargs=dict(dim=cfg.input_dim, dtype=self.dtype()),
            ),
        )
        self._add_child(
            "linear2",
            cfg.linear.set(
                create_module_kwargs=dict(features=cfg.output_dim),
                create_dummy_input_kwargs=dict(dim=cfg.hidden_dim, dtype=self.dtype()),
            ),
        )
        self._add_child(
            "norm",
            cfg.norm.set(
                create_module_kwargs={},
                create_dummy_input_kwargs=dict(dim=cfg.hidden_dim, dtype=self.dtype()),
            ),
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.norm(x, use_running_average=not self.is_training)
        x = nn.silu(x)
        x = self.linear2(x)
        return x


class FlaxLayerTest(TestCase):
    def testFeedForward(self):
        batch_size, seq_len, input_dim, hidden_dim, output_dim = 2, 5, 4, 8, 6
        cfg = FeedForward.default_config().set(
            name="test", input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
        )
        cfg.linear.vlog = 5
        layer: FeedForward = cfg.instantiate(parent=None)
        layer_params = layer.initialize_parameters_recursively(jax.random.PRNGKey(1))
        self.assertEqual(
            {
                "linear1": FrozenDict(
                    {
                        "params": {
                            "bias": (hidden_dim,),
                            "kernel": (input_dim, hidden_dim),
                        },
                    }
                ),
                "linear2": FrozenDict(
                    {
                        "params": {
                            "bias": (output_dim,),
                            "kernel": (hidden_dim, output_dim),
                        },
                    }
                ),
                "norm": FrozenDict(
                    {
                        "batch_stats": {
                            "mean": (hidden_dim,),
                            "var": (hidden_dim,),
                        },
                        "params": {
                            "bias": (hidden_dim,),
                            "scale": (hidden_dim,),
                        },
                    }
                ),
            },
            utils.shapes(layer_params),
        )

        inputs = jnp.ones([batch_size, seq_len, input_dim], dtype=jnp.float32)
        outputs, output_collection = F(
            layer,
            inputs=(inputs,),
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(0),
        )

        self.assertEqual((batch_size, seq_len, output_dim), utils.shapes(outputs))
        self.assertAlmostEqual(-4.712797453976236e-05, outputs.sum().item())
        self.assertEqual(
            [
                (
                    "state_updates/norm/batch_stats",
                    FrozenDict({"mean": (hidden_dim,), "var": (hidden_dim,)}),
                )
            ],
            utils.flatten_items(utils.shapes(output_collection)),
        )


if __name__ == "__main__":
    absltest.main()

from typing import Dict

import jax.random
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp

import param_init
from module import BaseLayer, ParameterPartitionSpec, ParameterSpec, PartitionSpec
from module import functional as F
from repeat import Repeat
from test_utils import TestCase, assert_allclose, shapes


class TestLayer(BaseLayer):
    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        return dict(
            inc=ParameterSpec(shape=[], partition=[], initializer=param_init.ConstantInitializer(1))
        )

    def init_forward_state(self, batch_size):
        return jnp.zeros([batch_size], dtype=self.dtype())

    def forward(self, carry, forward_state):
        logging.info("TestLayer: carry=%s forward_state=%s", shapes(carry), shapes(forward_state))
        self.add_summary("carry_mean", jnp.mean(carry))
        return carry + self.parameters["inc"], forward_state + carry


class TestRepeat(Repeat):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.layer = TestLayer.default_config()
        return cfg

    def init_forward_state(self, batch_size):
        cfg = self.config
        layer_state = self.layer.init_forward_state(batch_size)
        return dict(
            layer=jax.tree_map(
                lambda x: jnp.tile(x[None, :], [cfg.num_layers, 1]),
                layer_state,
            )
        )

    def forward(self, carry, forward_state):
        def fn(carry, forward_state_tn):
            return self.layer(carry, forward_state_tn["layer"])

        carry, forward_state = self._run(fn, carry, xs=forward_state)
        return carry, dict(layer=forward_state)


class RepeatTest(TestCase):
    @parameterized.parameters(jnp.float32, jnp.bfloat16)
    def testRepeat(self, dtype):
        batch_size, num_layers = 14, 4
        layer: TestRepeat = (
            TestRepeat.default_config()
            .set(name="test", num_layers=num_layers, dtype=dtype)
            .instantiate(parent=None)
        )
        self.assertEqual(
            {
                "layer": {
                    "inc": ParameterPartitionSpec(
                        shape=(num_layers,), partition=PartitionSpec(None), factorization=None
                    ),
                }
            },
            layer.create_partition_specs_recursively(),
        )
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(1))
        logging.info("layer params=%s", layer_params)

        input_forward_state = layer.init_forward_state(batch_size)
        (carry, output_forward_state), output_collection = F(
            layer,
            prng_key=jax.random.PRNGKey(2),
            state=layer_params,
            inputs=(jnp.arange(batch_size, dtype=dtype), input_forward_state),
            is_training=True,
        )
        logging.info("forward_state=%s", output_forward_state)
        logging.info("output_collection=%s", output_collection)
        assert_allclose(carry, jnp.arange(num_layers, num_layers + batch_size, dtype=dtype))
        self.assertEqual(shapes(input_forward_state), shapes(output_forward_state))
        assert_allclose(
            output_forward_state["layer"],
            jnp.reshape(
                jnp.arange(batch_size)[None, :] + jnp.arange(num_layers, dtype=dtype)[:, None],
                (num_layers, batch_size),
            ),
        )
        self.assertEqual(
            {"layer": {"carry_mean": (num_layers,)}},
            shapes(output_collection.summaries),
        )
        assert_allclose(
            0.5 * (batch_size - 1) + jnp.arange(num_layers, dtype=dtype),
            output_collection.summaries["layer"]["carry_mean"],
        )


if __name__ == "__main__":
    absltest.main()

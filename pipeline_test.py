from typing import Dict

import jax.random
from absl import logging
from absl.testing import absltest
from jax import numpy as jnp

import param_init
from module import BaseLayer, ParameterPartitionSpec, ParameterSpec, PartitionSpec
from module import functional as F
from pipeline import (
    Pipeline,
    transpose_from_pipeline_stage_outputs,
    transpose_to_pipeline_stage_inputs,
)
from test_utils import assert_allclose, shapes


class TransposeTest(absltest.TestCase):
    def testTransposeFunctions(self):
        num_layers, num_microbatches = 3, 5
        layer_indices = jnp.tile(jnp.arange(num_layers)[:, None], (1, num_microbatches))
        microbatch_indices = jnp.tile(jnp.arange(num_microbatches)[None, :], (num_layers, 1))
        # [num_layers, num_microbatches, 2].
        inputs = jnp.stack([layer_indices, microbatch_indices], axis=-1)

        # Transpose to pipeline inputs.
        transposed = transpose_to_pipeline_stage_inputs(inputs)
        logging.info("transposed=%s", transposed)
        for i in range(num_layers):
            for j in range(num_microbatches):
                t = i + j
                assert_allclose(jnp.asarray([i, j]), transposed[t, i])

        # Transpose from pipeline outputs.
        outputs = transpose_from_pipeline_stage_outputs(transposed)
        assert_allclose(outputs, inputs)


class TestLayer(BaseLayer):
    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        return dict(
            inc=ParameterSpec(shape=[], partition=[], initializer=param_init.ConstantInitializer(1))
        )

    def init_forward_state(self, batch_size):
        return jnp.zeros([batch_size])

    def forward(self, carry, forward_state):
        logging.info("TestLayer: carry=%s forward_state=%s", shapes(carry), shapes(forward_state))
        self.add_summary("carry_mean", jnp.mean(carry))
        return carry + self.parameters["inc"], forward_state + carry


class TestPipeline(Pipeline):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("microbatch_size", 0, "The microbatch size.")
        cfg.layer = TestLayer.default_config()
        return cfg

    def init_forward_state(self, batch_size):
        cfg = self.config
        layer_state = self.layer.init_forward_state(cfg.microbatch_size)
        num_microbatches = batch_size // cfg.microbatch_size
        return dict(
            layer=jax.tree_map(
                lambda x: jnp.tile(x[None, None, :], [cfg.num_layers, num_microbatches, 1]),
                layer_state,
            )
        )

    def forward(self, carry, forward_state):
        cfg = self.config
        carry = self._to_microbatches(carry, microbatch_size=cfg.microbatch_size)

        def fn(carry, forward_state_tn):
            return self.layer(carry, forward_state_tn["layer"])

        carry, forward_state = self._run(fn, carry, xs=forward_state)
        carry = self._from_microbatches(carry)
        return carry, dict(layer=forward_state)


class PipelineTest(absltest.TestCase):
    def testPipeline(self):
        batch_size, microbatch_size, num_layers = 14, 2, 4
        num_microbatches = batch_size // microbatch_size
        layer: TestPipeline = (
            TestPipeline.default_config()
            .set(name="test", num_layers=num_layers, microbatch_size=microbatch_size)
            .instantiate(parent=None)
        )
        self.assertEqual(
            {
                "layer": {
                    "inc": ParameterPartitionSpec(
                        shape=(num_layers,),
                        partition=PartitionSpec(
                            "pipeline",
                        ),
                        factorization=None,
                    )
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
            inputs=(jnp.arange(batch_size, dtype=jnp.float32), input_forward_state),
            is_training=True,
        )
        logging.info("forward_state=%s", output_forward_state)
        logging.info("output_collection=%s", output_collection)
        assert_allclose(carry, jnp.arange(num_layers, num_layers + batch_size))
        self.assertEqual(shapes(input_forward_state), shapes(output_forward_state))
        assert_allclose(
            output_forward_state["layer"],
            jnp.reshape(
                jnp.arange(batch_size)[None, :] + jnp.arange(num_layers)[:, None],
                (num_layers, num_microbatches, microbatch_size),
            ),
        )
        self.assertEqual(
            {"layer": {"carry_mean": (num_layers, num_microbatches)}},
            shapes(output_collection.summaries),
        )
        assert_allclose(
            0.5
            + jnp.arange(num_microbatches)[None, :] * microbatch_size
            + jnp.arange(num_layers)[:, None],
            output_collection.summaries["layer"]["carry_mean"],
        )


if __name__ == "__main__":
    absltest.main()

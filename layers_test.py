import copy
import math

import jax.random
import numpy as np
from absl.testing import absltest
from jax import numpy as jnp

import layers
from layers import Linear, LayerNorm, RMSNorm
import torch


def _assert_allclose(a, b, atol=1e-6, rtol=1e-3):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=np.abs(a - b).max())


def _shapes(nested_tensor):
    return jax.tree_map(lambda x: x.shape, nested_tensor)


def _copy(src: jnp.ndarray, dst: torch.nn.Parameter):
    with torch.no_grad():
        src = np.asarray(src).copy()
        src = torch.as_tensor(src)
        dst.copy_(src)


class LayerTest(absltest.TestCase):
    def testLinear(self):
        input_dim, output_dim = 4, 6
        cfg = Linear.default_config().set(
            name="test", input_dim=input_dim, output_dim=output_dim
        )
        layer: Linear = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        self.assertEqual(
            dict(weight=(input_dim, output_dim), bias=(output_dim,)),
            _shapes(layer_params),
        )
        bias = layer_params["bias"]
        _assert_allclose(bias, jnp.zeros_like(bias))
        # Randomize bias.
        layer_params["bias"] = jax.random.normal(
            jax.random.PRNGKey(45), shape=bias.shape, dtype=bias.dtype
        )

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [2, 3, input_dim])
        inputs = orig_inputs.copy()

        # Compute layer outputs.
        context = layer.make_invocation_context(
            is_training=True, parameters=layer_params, prng_key=prng_key
        )
        outputs = layer(inputs, context=context)

        # Compute ref outputs.
        ref = torch.nn.Linear(in_features=input_dim, out_features=output_dim)
        # torch.nn.Linear.weight is of shape (output_dim, input_dim).
        _copy(layer_params["weight"].transpose(), ref.weight)
        _copy(layer_params["bias"], ref.bias)
        ref_outputs = ref(torch.as_tensor(inputs))
        _assert_allclose(outputs, ref_outputs.detach().numpy())

    def testLayerNorm(self):
        dim = 6
        cfg = LayerNorm.default_config().set(name="norm", dim=dim)
        layer: LayerNorm = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [2, 3, dim])
        inputs = orig_inputs.copy()

        context = layer.make_invocation_context(
            is_training=True, parameters=layer_params, prng_key=prng_key
        )
        outputs = layer(inputs, context=context)
        # forward() should not mutate 'inputs' in-place.
        _assert_allclose(inputs, orig_inputs)
        # The output mean should be close to 0.
        output_mean = outputs.mean(axis=-1, keepdims=True)
        _assert_allclose(output_mean, np.zeros_like(output_mean))
        # The output variance should be close to 1.
        output_var = ((outputs - output_mean) ** 2).mean(axis=-1)
        _assert_allclose(output_var, np.ones_like(output_var))

        # Set scales to 2.
        layer_params2 = copy.deepcopy(layer_params)
        layer_params2["scale"] *= 2
        outputs = layer(inputs, context=context.clone(parameters=layer_params2))
        # The output mean should be close to 0.
        output_mean = outputs.mean(axis=-1, keepdims=True)
        _assert_allclose(output_mean, np.zeros_like(output_mean))
        # The output variance should be close to 4.
        output_var = ((outputs - output_mean) ** 2).mean(axis=-1)
        _assert_allclose(output_var, np.ones_like(output_var) * 4)

    def testRMSNorm(self):
        dim = 6
        cfg = RMSNorm.default_config().set(name="norm", dim=dim)
        layer: RMSNorm = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [2, 3, dim])
        inputs = orig_inputs.copy()

        context = layer.make_invocation_context(
            is_training=True, parameters=layer_params, prng_key=prng_key
        )
        outputs = layer(inputs, context=context)
        # forward() should not mutate 'inputs' in-place.
        _assert_allclose(inputs, orig_inputs)
        # The output_norm should be close to sqrt(dim).
        output_norm = jnp.sqrt((outputs ** 2).sum(axis=-1))
        _assert_allclose(output_norm, np.ones_like(output_norm) * math.sqrt(dim))

        # Set scales to 2.
        layer_params2 = copy.deepcopy(layer_params)
        layer_params2["scale"] *= 2
        outputs = layer(inputs, context=context.clone(parameters=layer_params2))
        output_norm = jnp.sqrt((outputs ** 2).sum(axis=-1))
        # The output_norm should be close to 2 * sqrt(dim).
        _assert_allclose(output_norm, np.ones_like(output_norm) * 2.0 * math.sqrt(dim))


if __name__ == "__main__":
    layers.enable_numeric_checks = True
    absltest.main()

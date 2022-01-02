import copy
import math

import jax.random
import numpy as np
from absl.testing import absltest
from jax import numpy as jnp

import layers
from layers import RMSNorm


def _assert_allclose(a, b, atol=1e-6, rtol=1e-3):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=np.abs(a - b).max())


class RMSNormTest(absltest.TestCase):

    def testNormalization(self):
        dim = 6
        cfg = RMSNorm.default_config().set(name='norm', dim=dim)
        layer: RMSNorm = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [2, 3, dim])
        inputs = orig_inputs.copy()

        context = layer.make_invocation_context(is_training=True, parameters=layer_params, prng_key=prng_key)
        outputs = layer(inputs, context=context)
        # forward() should not mutate 'inputs' in-place.
        _assert_allclose(inputs, orig_inputs)
        # The output_norm should be close to sqrt(dim).
        output_norm = jnp.sqrt((outputs ** 2).sum(axis=-1))
        _assert_allclose(output_norm, np.ones_like(output_norm) * math.sqrt(dim))

        # Set scales to 2.
        layer_params2 = copy.deepcopy(layer_params)
        layer_params2['scale'] *= 2
        outputs = layer(inputs, context=context.clone(parameters=layer_params2))
        output_norm = jnp.sqrt((outputs ** 2).sum(axis=-1))
        # The output_norm should be close to 2 * sqrt(dim).
        _assert_allclose(output_norm, np.ones_like(output_norm) * 2. * math.sqrt(dim))


if __name__ == "__main__":
    layers.enable_numeric_checks = True
    absltest.main()

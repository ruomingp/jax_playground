import math

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from param_init import DefaultInit


class DefaultInitTest(absltest.TestCase):
    def testBias(self):
        init: DefaultInit = DefaultInit.default_config().instantiate()
        bias = init.initialize(
            "bias", prng_key=jax.random.PRNGKey(1), shape=[4], dtype=jnp.float16
        )
        self.assertEqual(bias.dtype, jnp.float16)
        np.testing.assert_array_equal(bias, jnp.zeros_like(bias))

    def testScale(self):
        init: DefaultInit = DefaultInit.default_config().instantiate()
        scale = init.initialize(
            "scale", prng_key=jax.random.PRNGKey(1), shape=[4], dtype=jnp.bfloat16
        )
        self.assertEqual(scale.dtype, jnp.bfloat16)
        np.testing.assert_array_equal(scale, jnp.ones_like(scale))

    def testFan(self):
        shape = (4, 8, 3, 3)
        for dist in ("uniform", "normal"):
            for gain in (1.0, 2.0):
                for fan_type in ("fan_in", "fan_out", "xavier"):
                    init: DefaultInit = (
                        DefaultInit.default_config()
                        .set(fan=fan_type, gain=gain, distribution=dist)
                        .instantiate()
                    )
                    fan = init._get_fan("weight", shape)
                    self.assertEqual(
                        fan,
                        {
                            "fan_in": 8 * 3 * 3,
                            "fan_out": 4 * 3 * 3,
                            "xavier": 6 * 3 * 3,
                        }[fan_type],
                    )
                    weight = init.initialize(
                        "weight",
                        prng_key=jax.random.PRNGKey(1),
                        shape=shape,
                        dtype=jnp.float32,
                    )
                    self.assertEqual(weight.dtype, jnp.float32)
                    expected_std = gain / math.sqrt(fan)
                    actual_std = np.std(weight)
                    self.assertBetween(
                        actual_std, expected_std / 1.5, expected_std * 1.5
                    )


if __name__ == "__main__":
    absltest.main()

import math

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from param_init import DefaultInitializer, FanAxes, Shape


class DefaultInitTest(absltest.TestCase):
    def testBias(self):
        init: DefaultInitializer = DefaultInitializer.default_config().instantiate()
        bias = init.initialize("bias", prng_key=jax.random.PRNGKey(1), shape=[4], dtype=jnp.float16)
        self.assertEqual(bias.dtype, jnp.float16)
        np.testing.assert_array_equal(bias, jnp.zeros_like(bias))

    def testScale(self):
        init: DefaultInitializer = DefaultInitializer.default_config().instantiate()
        scale = init.initialize(
            "scale", prng_key=jax.random.PRNGKey(1), shape=[4], dtype=jnp.bfloat16
        )
        self.assertEqual(scale.dtype, jnp.bfloat16)
        np.testing.assert_array_equal(scale, jnp.ones_like(scale))

    def testFan(self):
        class CustomFanInitializer(DefaultInitializer):
            def _compute_fan_axes(self, name: str, shape: Shape) -> FanAxes:
                return FanAxes(in_axis=(0, 1), out_axis=2)

        shape = (3, 3, 8, 4)  # H, W, I, O
        init_classes = [DefaultInitializer, CustomFanInitializer]
        fan_specs = [
            {
                "fan_in": 8 * 3 * 3,
                "fan_out": 4 * 3 * 3,
                "fan_avg": 6 * 3 * 3,
            },
            {
                "fan_in": 3 * 3 * 4,
                "fan_out": 8 * 4,
                "fan_avg": 18 + 16,
            },
        ]
        for init_cls, fans in zip(init_classes, fan_specs):
            for dist in ("uniform", "normal", "truncated_normal"):
                for scale in (1.0, 2.0):
                    for fan_type in ("fan_in", "fan_out", "fan_avg"):
                        init = (
                            init_cls.default_config()
                            .set(fan=fan_type, scale=scale, distribution=dist)
                            .instantiate()
                        )
                        fan = fans[fan_type]
                        weight = init.initialize(
                            "weight",
                            prng_key=jax.random.PRNGKey(1),
                            shape=shape,
                            dtype=jnp.float32,
                        )
                        self.assertEqual(weight.dtype, jnp.float32)
                        expected_std = scale / math.sqrt(fan)
                        actual_std = np.std(weight)
                        self.assertBetween(actual_std, expected_std / 1.5, expected_std * 1.5)

    def testNoneFan(self):
        init: DefaultInitializer = (
            DefaultInitializer.default_config()
            .set(fan=None, scale=1.0, distribution="uniform")
            .instantiate()
        )
        weight_shape = [100, 100]
        weight = init.initialize(
            "weight", prng_key=jax.random.PRNGKey(1), shape=weight_shape, dtype=jnp.float32
        )
        std_err = 1 / (12 * np.sqrt(np.prod(weight_shape)))
        self.assertBetween(np.mean(weight), 0.5 - 6 * std_err, 0.5 + 6 * std_err)


if __name__ == "__main__":
    absltest.main()

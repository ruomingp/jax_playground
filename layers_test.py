import copy
import math

import jax.random
import numpy as np
import torch
from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp

from typing import Tuple, Union

import layers
import module
from module import functional as F, BaseLayer
from layers import Conv2D, Linear, LayerNorm, RMSNorm, BatchNorm


def _assert_allclose(a, b, atol=1e-6, rtol=1e-3):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=np.abs(a - b).max())


def _shapes(nested_tensor):
    return jax.tree_map(lambda x: tuple(x.shape), nested_tensor)


def _as_torch_tensor(src: jnp.ndarray):
    return torch.as_tensor(np.asarray(src).copy())


def _copy(src: jnp.ndarray, dst: torch.nn.Parameter):
    with torch.no_grad():
        src = np.asarray(src).copy()
        src = torch.as_tensor(src)
        dst.copy_(src)


def _initialize_parameters_recursively(
        prng_key: jax.random.KeyArray, layer: BaseLayer
):
    param_specs = layer.create_parameter_specs_recursively()
    params = layer.initialize_parameters_recursively(prng_key, param_specs)
    return param_specs, params


class LayerTest(parameterized.TestCase):
    def testLayerNorm(self):
        dim = 6
        cfg = LayerNorm.default_config().set(name="norm", dim=dim)
        layer: LayerNorm = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_param_specs, layer_params = _initialize_parameters_recursively(
            init_key, layer
        )
        self.assertEqual(dict(scale=(dim,), bias=(dim,)), _shapes(layer_param_specs))
        self.assertEqual(_shapes(layer_param_specs), _shapes(layer_params))

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [2, 3, dim])
        inputs = orig_inputs.copy()

        outputs, _ = F(layer, inputs=(inputs,), is_training=True, state=layer_params, prng_key=prng_key)
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
        outputs, _ = F(layer, inputs=(inputs,), is_training=True, state=layer_params2, prng_key=prng_key)
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
        layer_param_specs, layer_params = _initialize_parameters_recursively(
            init_key, layer
        )
        self.assertEqual(dict(scale=(dim,)), _shapes(layer_param_specs))
        self.assertEqual(_shapes(layer_param_specs), _shapes(layer_params))

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [2, 3, dim])
        inputs = orig_inputs.copy()

        outputs, _ = F(layer, inputs=(inputs,), is_training=True, state=layer_params, prng_key=prng_key)
        # forward() should not mutate 'inputs' in-place.
        _assert_allclose(inputs, orig_inputs)
        # The output_norm should be close to sqrt(dim).
        output_norm = jnp.sqrt((outputs ** 2).sum(axis=-1))
        _assert_allclose(output_norm, np.ones_like(output_norm) * math.sqrt(dim))

        # Set scales to 2.
        layer_params2 = copy.deepcopy(layer_params)
        layer_params2["scale"] *= 2
        outputs, _ = F(layer, inputs=(inputs,), is_training=True, state=layer_params2, prng_key=prng_key)
        output_norm = jnp.sqrt((outputs ** 2).sum(axis=-1))
        # The output_norm should be close to 2 * sqrt(dim).
        _assert_allclose(output_norm, np.ones_like(output_norm) * 2.0 * math.sqrt(dim))

    def testBatchNorm(self):
        dim = 6
        cfg = BatchNorm.default_config().set(name="norm", dim=dim)
        layer: BatchNorm = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_param_specs, layer_params = _initialize_parameters_recursively(
            init_key, layer
        )
        self.assertEqual(
            dict(scale=(dim,), bias=(dim,), moving_mean=(dim,), moving_variance=(dim,)),
            _shapes(layer_param_specs),
        )
        self.assertEqual(_shapes(layer_param_specs), _shapes(layer_params))

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        orig_inputs = jax.random.normal(input_key, [2, 3, dim])
        inputs = orig_inputs.copy()

        for is_training in (True, False):
            outputs, output_collection = F(layer, inputs=(inputs,), is_training=is_training, state=layer_params,
                                           prng_key=prng_key,
                                           output_collection_sections=[module.OutputCollection.SECTION_STATE_UPDATE])
            param_updates = output_collection[module.OutputCollection.SECTION_STATE_UPDATE]
            if is_training:
                # The output mean should be close to 0.
                output_mean = jnp.mean(outputs, axis=(0, 1), keepdims=True)
                _assert_allclose(output_mean, np.zeros_like(output_mean))
                # The output variance should be close to 1.
                output_var = jnp.mean((outputs - output_mean) ** 2, axis=(0, 1))
                _assert_allclose(output_var, np.ones_like(output_var))
                # Check parameter updates.
                self.assertCountEqual(
                    ["moving_mean", "moving_variance"], param_updates.keys()
                )
                self.assertNotAlmostEqual(
                    jnp.abs(
                        param_updates["moving_mean"] - layer_params["moving_mean"]
                    ).max(),
                    0,
                )
                self.assertNotAlmostEqual(
                    jnp.abs(
                        param_updates["moving_variance"]
                        - layer_params["moving_variance"]
                    ).max(),
                    0,
                )
            else:
                _assert_allclose(outputs, inputs)
                self.assertCountEqual([], param_updates.keys())

    def testLinear(self):
        input_dim, output_dim = 4, 6
        cfg = Linear.default_config().set(
            name="test", input_dim=input_dim, output_dim=output_dim
        )
        layer: Linear = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_param_specs, layer_params = _initialize_parameters_recursively(
            init_key, layer
        )
        self.assertEqual(
            dict(weight=(input_dim, output_dim), bias=(output_dim,)),
            _shapes(layer_param_specs),
        )
        self.assertEqual(_shapes(layer_param_specs), _shapes(layer_params))
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
        outputs, _ = F(layer, inputs=(inputs,), is_training=True, state=layer_params, prng_key=prng_key)

        # Compute ref outputs.
        ref = torch.nn.Linear(in_features=input_dim, out_features=output_dim)
        # torch.nn.Linear.weight is of shape (output_dim, input_dim).
        _copy(layer_params["weight"].transpose(), ref.weight)
        _copy(layer_params["bias"], ref.bias)
        ref_outputs = ref(torch.as_tensor(inputs))
        _assert_allclose(outputs, ref_outputs.detach().numpy())

    @parameterized.named_parameters(
        ("1x1", (1, 1), (1, 1), "VALID"),
        ("2x2_VALID", (2, 2), (1, 1), "VALID"),
        ("2x2_SAME", (2, 2), (1, 1), "SAME"),
        ("2x2_S2_VALID", (2, 2), (2, 2), "VALID"),
        ("3x3_VALID", (3, 3), (1, 1), "VALID"),
        ("3x3_SAME", (3, 3), (1, 1), "SAME"),
        ("3x3_S2_VALID", (3, 3), (2, 2), "VALID"),
        ("3x3_S2_PADDING1", (3, 3), (2, 2), (1, 1)),
    )
    def testConv2D(
            self,
            window: Tuple[int, int],
            strides: Tuple[int, int],
            padding: Union[str, Tuple[int, int]],
    ):
        input_dim, output_dim = 4, 6
        if isinstance(padding, tuple):
            conv_padding = ((padding[0], padding[0]), (padding[1], padding[1]))
        else:
            conv_padding = padding
        cfg = Conv2D.default_config().set(
            name="test",
            input_dim=input_dim,
            output_dim=output_dim,
            window=window,
            strides=strides,
            padding=conv_padding,
        )
        layer: Conv2D = cfg.instantiate(parent=None)

        # Initialize layer parameters.
        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        layer_param_specs, layer_params = _initialize_parameters_recursively(
            init_key, layer
        )
        self.assertEqual(
            dict(
                weight=(window[0], window[1], input_dim, output_dim), bias=(output_dim,)
            ),
            _shapes(layer_param_specs),
        )
        self.assertEqual(_shapes(layer_param_specs), _shapes(layer_params))
        bias = layer_params["bias"]
        _assert_allclose(bias, jnp.zeros_like(bias))
        # Randomize bias.
        layer_params["bias"] = jax.random.normal(
            jax.random.PRNGKey(45), shape=bias.shape, dtype=bias.dtype
        )

        # Random inputs.
        prng_key, input_key = jax.random.split(prng_key)
        inputs = jax.random.normal(input_key, [2, 7, 7, input_dim])

        # Compute layer outputs.
        outputs, _ = F(layer, inputs=(inputs,), is_training=True, state=layer_params, prng_key=prng_key)

        # Compute ref outputs.
        ref_padding = padding.lower() if isinstance(padding, str) else padding
        ref = torch.nn.Conv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=window,
            stride=strides,
            padding=ref_padding,
        )
        # torch.nn.Linear.weight is of shape (output_dim, input_dim, kernel_size...).
        _copy(layer_params["weight"].transpose(3, 2, 0, 1), ref.weight)
        _copy(layer_params["bias"], ref.bias)
        ref_outputs = ref(_as_torch_tensor(inputs.transpose(0, 3, 1, 2)))
        _assert_allclose(outputs, ref_outputs.detach().numpy().transpose(0, 2, 3, 1))


if __name__ == "__main__":
    layers.enable_numeric_checks = True
    absltest.main()

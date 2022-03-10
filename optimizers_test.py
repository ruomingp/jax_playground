import jax.nn
import numpy as np
import optax
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from module import ParameterPartitionSpec, PartitionSpec, Tensor
from optimizer_base import OptParam, OptStatePartitionSpec
from optimizers import (
    adafactor_optimizer,
    adamw_optimizer,
    clip_by_block_rms,
    clip_by_global_norm,
    scale_by_param_block_rms,
    sgd_optimizer,
)
from test_utils import TestCase, assert_allclose
from utils import VDict, shapes


def rms_norm(x):
    return jnp.sqrt(jnp.mean(x**2))


class OptimizerTest(TestCase):
    @parameterized.parameters((0.1, 0), (0.1, 0.01))
    def testSGD(self, learning_rate, weight_decay):
        sgd = sgd_optimizer(learning_rate=learning_rate, weight_decay=weight_decay)
        params = OptParam(
            value=jnp.asarray([0, 1, 2, -3], dtype=jnp.float32), factorization_spec=None
        )
        state = sgd.init(params)

        def loss(x):
            return -jax.nn.log_softmax(x)[1]

        loss, grads = jax.value_and_grad(loss)(params.value)
        np.testing.assert_allclose(loss, 1.412078, atol=1e-6)
        np.testing.assert_allclose(grads, [0.089629, -0.756364, 0.662272, 0.004462], atol=1e-6)

        updates, updated_state = sgd.update(grads, state=state, params=params)
        np.testing.assert_allclose(
            updates, -learning_rate * (grads + weight_decay * params.value), atol=1e-6
        )

        updated_params = optax.apply_updates(params.value, updates)
        np.testing.assert_allclose(updated_params, params.value + updates, atol=1e-6)

    @parameterized.parameters((0.1, 0), (0.1, 0.01))
    def testAdamW(self, learning_rate, weight_decay):
        self._test_optimizer(
            adamw_optimizer(learning_rate=learning_rate, weight_decay=weight_decay)
        )

    @parameterized.parameters((0.1, 0), (0.1, 0.01))
    def testAdafactor(self, learning_rate, weight_decay):
        self._test_optimizer(
            adafactor_optimizer(
                learning_rate=learning_rate, momentum=0.9, weight_decay_rate=weight_decay
            )
        )

    def _test_optimizer(self, optimizer):
        params = OptParam(
            value=jnp.asarray([0, 1, 2, -3], dtype=jnp.float32), factorization_spec=None
        )
        state = optimizer.init(params)

        param_partition_spec = ParameterPartitionSpec(
            shape=[4], partition=PartitionSpec("model"), factorization=None
        )
        state_partition_spec = optimizer.partition(param_partition_spec)
        logging.info("state_partition_spec=%s state=%s", state_partition_spec, shapes(state))

        def check_partition_spec(spec: OptStatePartitionSpec, tree):
            if spec.partition is None:
                return
            self.assertIsInstance(tree, Tensor)
            self.assertEqual(list(spec.shape), list(tree.shape))
            self.assertEqual(len(spec.partition), tree.ndim)

        jax.tree_map(check_partition_spec, state_partition_spec, state)

        def compute_loss(x):
            return -jax.nn.log_softmax(x)[1]

        loss, grads = jax.value_and_grad(compute_loss)(params.value)
        updates, updated_state = optimizer.update(grads, state=state, params=params)
        updated_params = optax.apply_updates(params.value, updates)
        new_loss = compute_loss(updated_params)
        self.assertLess(new_loss, loss)

    @parameterized.parameters(None, 100.0, 0.1)
    def testGradientClipping(self, max_norm):
        clip = clip_by_global_norm(max_norm=max_norm)
        params = jnp.asarray([0, 1, 2, -3], dtype=jnp.float32)
        state = clip.init(params)

        def loss(x):
            return -jax.nn.log_softmax(x)[1]

        loss, grads = jax.value_and_grad(loss)(params)
        np.testing.assert_allclose(loss, 1.412078, atol=1e-6)
        np.testing.assert_allclose(grads, [0.089629, -0.756364, 0.662272, 0.004462], atol=1e-6)
        g_norm = optax.global_norm(grads)

        updates, updated_state = clip.update(grads, state=state, params=params)
        if max_norm is None or g_norm < max_norm:
            np.testing.assert_allclose(updates, grads, atol=1e-6)
        else:
            np.testing.assert_allclose(max_norm, optax.global_norm(updates))

    @parameterized.parameters(100.0, 1e-3)
    def testClipByBlockRMS(self, max_norm):
        clip = clip_by_block_rms(threshold=max_norm)
        params = VDict(x=jnp.asarray([[0, 0, 0, 0], [0, 1, 2, -3]], dtype=jnp.float32))
        state = clip.init(params)
        self.assertEqual(optax.EmptyState, type(state))

        def loss(params):
            return -jax.nn.log_softmax(params["x"])[:, 1].mean()

        loss, grads = jax.value_and_grad(loss)(params)
        assert_allclose(loss, 1.399186)
        assert_allclose(
            [[0.125, -0.375, 0.125, 0.125], [0.044814, -0.378182, 0.331136, 0.002231]], grads["x"]
        )

        g_norm = jax.vmap(rms_norm)(grads["x"])
        assert_allclose([0.216506, 0.252332], g_norm)

        updates, updated_state = clip.update(grads, state=state, params=params)
        if max_norm > 1:
            np.testing.assert_allclose(updates["x"], grads["x"], atol=1e-6)
        else:
            np.testing.assert_allclose(jax.vmap(rms_norm)(updates["x"]), [max_norm] * 2)

    @parameterized.parameters(100.0, 1e-3)
    def testScaleByParamBlockRMS(self, threshold):
        scale = scale_by_param_block_rms(threshold)
        params = VDict(x=jnp.asarray([[0, 0, 0, 0], [0, 1, 2, -3]], dtype=jnp.float32))
        p_norm = jax.vmap(rms_norm)(params["x"])
        state = scale.init(params)
        self.assertEqual(optax.EmptyState, type(state))

        grads = VDict(x=jnp.asarray([[1e-5] * 4, [1] * 4], dtype=jnp.float32))

        g_norm = jax.vmap(rms_norm)(grads["x"])
        assert_allclose([1e-5, 1.0], g_norm)

        updates, updated_state = scale.update(grads, state=state, params=params)
        np.testing.assert_allclose(
            jax.vmap(rms_norm)(updates["x"]), jnp.maximum(p_norm, threshold) * g_norm
        )


if __name__ == "__main__":
    absltest.main()

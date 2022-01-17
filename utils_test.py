import jax
import numpy as np
import tensorflow as tf
import torch
from absl.testing import absltest
from jax import numpy as jnp

from utils import as_tensor, flatten_items, tree_paths


class TreeUtilsTest(absltest.TestCase):
    def testTreePaths(self):
        tree = {"a": 1, "b": [2, {"c": 3}]}
        self.assertEqual({"a": "a", "b": ["b/0", {"c": "b/1/c"}]}, tree_paths(tree))

    def testFlattenItems(self):
        tree = {"a": 1, "b": [2, {"c": 3, "d": 4}]}
        self.assertEqual(
            [("a", 1), ("b/0", 2), ("b/1/c", 3), ("b/1/d", 4)], flatten_items(tree)
        )
        self.assertEqual(
            [("a", 1), ("b.0", 2), ("b.1.c", 3), ("b.1.d", 4)],
            flatten_items(tree, separator="."),
        )

    def assertTensorEqual(self, a, b):
        self.assertIsInstance(a, jnp.ndarray)
        self.assertIsInstance(b, jnp.ndarray)
        self.assertEqual(a.dtype, b.dtype)
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_array_equal(a, b)

    def testAsTensor(self):
        # From a number.
        self.assertTensorEqual(jnp.ones([], dtype=jnp.int32), as_tensor(1))
        # From a numpy array.
        self.assertTensorEqual(
            jnp.ones([2], dtype=jnp.float32), as_tensor(np.ones([2], dtype=np.float32))
        )
        # From a TF tensor.
        self.assertTensorEqual(
            jnp.ones([3], dtype=jnp.bfloat16),
            as_tensor(tf.ones([3], dtype=tf.bfloat16)),
        )
        # From a Torch tensor.
        self.assertTensorEqual(
            jnp.ones([4, 1], dtype=jnp.float16),
            as_tensor(torch.ones([4, 1], dtype=torch.float16)),
        )
        # From a nested structure.
        jax.tree_map(
            self.assertTensorEqual,
            {
                "a": jnp.ones([1], dtype=jnp.float32),
                "b": [jnp.asarray([2]), {"c": jnp.asarray([[4]])}],
            },
            as_tensor(
                {
                    "a": np.ones([1], dtype=np.float32),
                    "b": [torch.as_tensor([2]), {"c": tf.convert_to_tensor([[4]])}],
                }
            ),
        )


if __name__ == "__main__":
    absltest.main()

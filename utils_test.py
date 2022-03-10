from collections import OrderedDict
from typing import Any, NamedTuple

import jax
import numpy as np
import tensorflow as tf
import torch
from absl.testing import absltest
from flax import serialization
from jax import numpy as jnp

from test_utils import TestCase
from utils import VDict, as_tensor, flatten_items, shapes, tree_paths, vectorized_tree_map


class Combo(NamedTuple):
    head: Any
    tail: Any


class TreeUtilsTest(TestCase):
    def testTreePaths(self):
        tree = {"a": 1, "b": [2, {"c": 3}]}
        self.assertEqual({"a": "a", "b": ["b/0", {"c": "b/1/c"}]}, tree_paths(tree))

        # Tuple.
        self.assertEqual(("0", ("1/0", "1/1"), "2"), tree_paths(("a", ("b", "c"), "d")))

        # NamedTuple.
        self.assertEqual(
            Combo(head="head", tail=Combo(head="tail/head", tail="tail/tail")),
            tree_paths(Combo(head=1, tail=Combo(head=2, tail=None))),
        )

    def testFlattenItems(self):
        tree = {"a": 1, "b": [2, {"c": 3, "d": 4}]}
        self.assertEqual([("a", 1), ("b/0", 2), ("b/1/c", 3), ("b/1/d", 4)], flatten_items(tree))
        self.assertEqual(
            [("a", 1), ("b.0", 2), ("b.1.c", 3), ("b.1.d", 4)],
            flatten_items(tree, separator="."),
        )
        kv = [("a", 1), ("b", 2)]
        d1 = OrderedDict(kv)
        d2 = OrderedDict(reversed(kv))
        self.assertEqual([("a", 1), ("b", 2)], sorted(flatten_items(d1)))
        self.assertEqual([("a", 1), ("b", 2)], sorted(flatten_items(d2)))

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

    def testVectorizedTreeMap(self):
        tree = VDict(a=jnp.arange(10), b=jnp.arange(7) - 3)
        self.assertEqual(VDict(a="a", b="b"), tree_paths(tree))
        self.assertNestedAllClose([("a", tree["a"]), ("b", tree["b"])], flatten_items(tree))

        # Stack 3 trees together.
        stacked_tree = jax.tree_map(lambda *xs: jnp.stack(xs), tree, tree, tree)
        self.assertEqual(type(stacked_tree), VDict)
        self.assertEqual(VDict(a=(3, 10), b=(3, 7)), jax.tree_map(lambda t: t.shape, stacked_tree))

        # jax.tree_map() treats VDict similarly to dict.
        self.assertEqual(VDict(a=45 * 3, b=0), jax.tree_map(lambda t: t.sum(), stacked_tree))
        # vectorized_tree_map() vectorizes 'fn' on VDict and processes the 3 trees separately.
        self.assertNestedAllClose(
            VDict(a=jnp.asarray([45, 45, 45]), b=jnp.asarray([0, 0, 0])),
            vectorized_tree_map(lambda t: t.sum(), stacked_tree),
        )

        # Nested VDict.
        tree2 = VDict(c=stacked_tree)
        stacked_tree2 = jax.tree_map(lambda *xs: jnp.stack(xs), tree2, tree2)
        self.assertEqual(
            VDict(c=VDict(a=(2, 3, 10), b=(2, 3, 7))),
            jax.tree_map(lambda t: t.shape, stacked_tree2),
        )
        self.assertNestedAllClose(
            VDict(c=VDict(a=jnp.full([2, 3], 45), b=jnp.full([2, 3], 0))),
            vectorized_tree_map(lambda t: t.sum(), stacked_tree2),
        )

    def testVDictSerialization(self):
        state_dict = dict(a=jnp.arange(10), b=jnp.arange(7) - 3)
        tree = VDict(**state_dict)
        v_state_dict = serialization.to_state_dict(tree)
        self.assertEqual(v_state_dict, state_dict)
        new_tree = serialization.from_state_dict(VDict, state=v_state_dict)
        self.assertEqual(new_tree, tree)
        # Check if `to_bytes` works as expected.
        self.assertEqual(serialization.to_bytes(state_dict), serialization.to_bytes(tree))


if __name__ == "__main__":
    absltest.main()

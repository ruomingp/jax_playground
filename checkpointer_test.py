import tempfile

import numpy as np
from absl import logging
from absl.testing import absltest
from jax import numpy as jnp

import utils
from checkpointer import Checkpointer


class CheckpointerTest(absltest.TestCase):
    def assertNestedEqual(self, a, b):
        a_kv = utils.flatten_items(a)
        b_kv = utils.flatten_items(b)
        self.assertCountEqual([k for k, _ in a_kv], [k for k, _ in b_kv])
        a_dict = dict(a_kv)
        b_dict = dict(b_kv)
        for k in a_dict:
            np.testing.assert_array_equal(a_dict[k], b_dict[k], err_msg=k)

    def testSaveAndRestore(self):
        cfg = Checkpointer.default_config().set(name="test", dir=tempfile.mkdtemp())
        ckpt: Checkpointer = cfg.instantiate(parent=None)
        state0 = dict(x=jnp.zeros([], dtype=jnp.int32), y=jnp.ones([2], dtype=jnp.float32))
        state1 = dict(x=jnp.ones([], dtype=jnp.int32), y=jnp.ones([2], dtype=jnp.float32) + 1)

        # Restoring from an empty dir returns the input state if step=None.
        self.assertNestedEqual((None, state0), ckpt.restore(step=None, state=state0))
        self.assertNestedEqual((None, state1), ckpt.restore(step=None, state=state1))
        # With an explicit step, ValueError will be raised.
        with self.assertRaises(ValueError):
            ckpt.restore(step=0, state=state0)

        ckpt.save(step=0, state=state0)
        self.assertNestedEqual((0, state0), ckpt.restore(step=0, state=state1))
        # step=None restores from the latest ckpt.
        self.assertNestedEqual((0, state0), ckpt.restore(step=None, state=state1))

        ckpt.save(step=1, state=state1)
        self.assertNestedEqual((1, state1), ckpt.restore(step=1, state=state0))
        # step=None restores from the latest ckpt.
        self.assertNestedEqual((1, state1), ckpt.restore(step=None, state=state0))

        # With state=None, we don't perform checks on the structure, dtypes, and shapes.
        self.assertNestedEqual((1, state1), ckpt.restore())

        # When the given state has a different dict key: 'z' instead of 'y'.
        with self.assertRaisesRegex(KeyError, "z"):
            ckpt.restore(
                step=None,
                state=dict(x=jnp.zeros([], dtype=jnp.int32), z=jnp.ones([2], dtype=jnp.float32)),
            )

        # When the given state has a different array shape: [3] instead of [2] for y.
        with self.assertRaisesRegex(ValueError, "checkpoint tree dtypes or shapes"):
            ckpt.restore(
                step=None,
                state=dict(x=jnp.zeros([], dtype=jnp.int32), y=jnp.ones([3], dtype=jnp.float32)),
            )

        # When the given state has a different dict shape: [1] instead of [] for x.
        with self.assertRaisesRegex(ValueError, "checkpoint tree dtypes or shapes"):
            ckpt.restore(
                step=None,
                state=dict(
                    x=jnp.zeros([1], dtype=jnp.int32),
                    y=jnp.ones([2], dtype=jnp.float32),
                ),
            )

        # When the given state has a different dtype: float32 instead of int32 for x.
        with self.assertRaisesRegex(ValueError, "checkpoint tree dtypes or shapes"):
            ckpt.restore(
                step=None,
                state=dict(
                    x=jnp.zeros([], dtype=jnp.float32),
                    y=jnp.ones([2], dtype=jnp.float32),
                ),
            )


if __name__ == "__main__":
    absltest.main()

"""A simple checkpointer based on flax.training.checkpoints.

It provides additional guards on top of the flax library to verify that dtypes and shapes match those of the model
parameters.
"""

import difflib
from typing import List, Optional

import jax
from flax.training import checkpoints as flax_checkpoints

import utils
from module import Module, NestedTensor


class Checkpointer(Module):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("dir", None, "The output directory.")
        cfg.define("write_every_n_steps", 1, "Writes checkpoint every N steps.")
        cfg.define("keep_last_n", 1, "Keeps this many past ckpts.")
        cfg.define(
            "keep_every_n_steps",
            None,
            "If > 0, keeps at least one checkpoint every N steps.",
        )
        return cfg

    @property
    def ckpt_dir(self):
        return f"{self.config.dir}/process_{jax.process_index():03d}"

    def save(self, *, step: int, state: NestedTensor):
        """Saves `state` at the given `step`."""
        cfg = self.config
        if step % cfg.write_every_n_steps != 0:
            return
        flax_checkpoints.save_checkpoint(
            ckpt_dir=self.ckpt_dir,
            step=step,
            target=self._checkpoint_target(state),
            keep=cfg.keep_last_n,
            keep_every_n_steps=cfg.keep_every_n_steps,
        )

    def restore(
        self, *, step: Optional[int] = None, state: Optional[NestedTensor] = None
    ) -> NestedTensor:
        """Restores from the checkpoint directory.

        Args:
            step: if None, restores from the latest checkpoint. Otherwise from the specified step.
            state: if not None, ensures that the restored state have the same structure, dtypes, and shapes as `state`.

        Returns:
            The restored checkpoint state. If step is None and no checkpoint is found, returns the input `state`.
        """
        input_target = self._checkpoint_target(state)
        restored_target = flax_checkpoints.restore_checkpoint(
            ckpt_dir=self.ckpt_dir, target=input_target, step=step
        )
        if state is not None:
            diff = self._diff(
                restored_target["dtypes_shapes"], input_target["dtypes_shapes"]
            )
            if diff:
                raise ValueError(
                    "Unable to restore checkpoint. A mismatch between the saved "
                    "checkpoint tree dtypes or shapes and the current one has been detected:\n"
                    f"{diff}"
                )
        return restored_target["state"]

    def _checkpoint_target(self, state: NestedTensor):
        if state is None:
            return None
        return {
            "state": state,
            "dtypes_shapes": self._dtypes_shapes(state),
        }

    def _dtypes_shapes(self, state: NestedTensor):
        paths = utils.tree_paths(state)
        dtypes = jax.tree_map(lambda x: x.dtype, state)
        shapes = jax.tree_map(lambda x: x.shape, state)
        items = []
        jax.tree_map(
            lambda path, dtype, shape: items.append(f"{path}={dtype}{shape}"),
            paths,
            dtypes,
            shapes,
        )
        return items

    def _diff(self, a: List[str], b: List[str]):
        if a == b:
            return None
        return "\n".join(difflib.ndiff(a, b))

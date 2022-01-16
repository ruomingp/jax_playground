import difflib
from typing import Any, Dict, List, Optional

import jax
from flax.training import checkpoints as flax_checkpoints

import config as config_lib
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

    def _dtypes_shapes(self, state: NestedTensor):
        paths = utils.tree_paths(state)
        dtypes = jax.tree_map(lambda x: x.dtype, state)
        shapes = jax.tree_map(lambda x: x.shape, state)
        items = []
        jax.tree_map(lambda path, dtype, shape: items.append(f"{path}={dtype}{shape}"), paths, dtypes, shapes)
        return items

    def _checkpoint_target(self, state: NestedTensor):
        flattened_state, _ = jax.tree_flatten(state)
        return {
            # Extract/flatten data structure to store to disk. Flax requires a flattened
            # data structure to be passed to the checkpointer.
            "flattened_state": flattened_state,
            "dtypes_shapes": self._dtypes_shapes(state),
        }

    def save(self, *, step: int, state: NestedTensor):
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
        # TODO(ruoming): add synchronization across processes.

    def _diff(self, a: List[str], b: List[str]):
        if a == b:
            return None
        return '\n'.join(difflib.ndiff(a, b))

    def restore(self, *, step: Optional[int] = None, state: NestedTensor) -> NestedTensor:
        input_target = self._checkpoint_target(state)
        restored_target = flax_checkpoints.restore_checkpoint(
            ckpt_dir=self.ckpt_dir, target=input_target, step=step
        )
        diff = self._diff(restored_target["dtypes_shapes"], input_target["dtypes_shapes"])
        if diff:
            raise ValueError(
                "Unable to restore checkpoint. A mismatch between the saved "
                "checkpoint tree structure, dtypes, or shapes and the current one has been detected:\n"
                f"{diff}"
            )
        return jax.tree_unflatten(jax.tree_structure(state), restored_target["flattened_state"])

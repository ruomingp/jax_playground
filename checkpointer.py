from typing import Any, Dict, Optional

from absl import logging
import jax
from flax.training import checkpoints as flax_checkpoints

import config as config_lib
from module import Module, tree_paths


class Checkpointer(Module):

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define('dir', None, 'The output directory.')
        cfg.define('write_every_n_steps', 1, 'Writes checkpoint every N steps.')
        cfg.define('keep_last_n', 1, 'Keeps this many past ckpts.')
        cfg.define('keep_every_n_steps', None, 'If > 0, keeps at least one checkpoint every N steps.')
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config

    @property
    def ckpt_dir(self):
        return f"{self.config.dir}/process_{jax.process_index():03d}"

    def save(self, step: int, state: Dict[str, Any]):
        cfg = self.config
        if step % cfg.write_every_n_steps != 0:
            return
        # Extract/flatten data structure to store to disk. Flax requires a flattened
        # data structure to be passed to the checkpointer.
        flattened_state, pytree_state = jax.tree_flatten(state)
        flattened_paths, _ = jax.tree_flatten(tree_paths(state))
        for path, entry in zip(flattened_paths, flattened_state):
            logging.info("flattened_state: %s=%s", path, type(entry))
        checkpoint_target = {
            'flattened_state': flattened_state,
            # Saves a serialized version of the pytree structure to detect potential
            # mismatch caused by different versions of saver/restorer.
            'str_pytree_state': str(pytree_state),
        }
        flax_checkpoints.save_checkpoint(
            ckpt_dir=self.ckpt_dir, step=step, target=checkpoint_target, keep=cfg.keep_last_n,
            keep_every_n_steps=cfg.keep_every_n_steps)
        # TODO(ruoming): add synchronization across processes.

    def restore(self, state, step: Optional[int] = None) -> Dict[str, Any]:
        flattened_state, pytree_state = jax.tree_flatten(state)
        str_pytree_state = str(pytree_state)
        input_target = {
            'flattened_state': flattened_state,
            'str_pytree_state': str_pytree_state,
        }
        restored_target = flax_checkpoints.restore_checkpoint(ckpt_dir=self.ckpt_dir, target=input_target, step=step)
        restored_state = restored_target['flattened_state']
        restored_str_pytree_state = restored_target['str_pytree_state']
        if restored_str_pytree_state != str_pytree_state:
            raise ValueError(
                'Unable to restore checkpoint. A mismatch between the saved '
                'checkpoint structure and the current one has been detected '
                f'(`{restored_str_pytree_state}` vs `{str_pytree_state}`).')
        return jax.tree_unflatten(pytree_state, restored_state)



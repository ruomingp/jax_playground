import jax
from jax import numpy as jnp

import config as config_lib
from module import Module

from typing import Any, Dict, Mapping, Optional, Sequence, Union
Tensor = jnp.ndarray
from flax.training import checkpoints as flax_checkpoints


class Checkpointer(Module):

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define('dir', None, 'The output directory.')
        cfg.define('write_every_n_steps', 1, 'Writes checkpoint every N steps.')
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        flax_checkpoints.save_checkpoint()

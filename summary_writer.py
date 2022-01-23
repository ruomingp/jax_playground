import numbers
from typing import Any, Dict, Optional

import guppy
import jax
from absl import logging
from jax import numpy as jnp
from tensorflow import summary as tf_summary

import config as config_lib
from metrics import WeightedScalar
from module import Module
from utils import tree_paths

Tensor = jnp.ndarray


class SummaryWriter(Module):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("dir", None, "The output directory.")
        cfg.define("write_every_n_steps", 1, "Writes summary every N steps.")
        cfg.define("print_heap", False, "Whether print the heap profile when writing the summary.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self.summary_writer: tf_summary.SummaryWriter = (
            tf_summary.create_file_writer(cfg.dir)
            if jax.process_index() == 0
            else tf_summary.create_noop_writer()
        )

    def __call__(self, step: int, values: Dict[str, Any]):
        cfg = self.config
        if step % cfg.write_every_n_steps != 0:
            return
        with self.summary_writer.as_default(step=step):
            values = jax.tree_map(
                lambda v: v.mean if isinstance(v, WeightedScalar) else v,
                values,
                is_leaf=lambda x: isinstance(x, WeightedScalar),
            )

            def write(path: str, value: jnp.ndarray):
                logging.info("SummaryWriter %s: %s=%s", self.path(), path, value)
                if isinstance(value, WeightedScalar):
                    tf_summary.scalar(path, value.mean, step=step)
                elif isinstance(value, numbers.Number) or value.ndim == 0:
                    tf_summary.scalar(path, value, step=step)
                else:
                    tf_summary.histogram(path, value, step=step)

            jax.tree_map(write, tree_paths(values, separator="/"), values)
            self.summary_writer.flush()

        if cfg.print_heap:
            heap = guppy.hpy().heap()
            print(heap[0])
            print(heap[1])
            print(heap[2])

"""Tensorflow Dataset Inputs."""
from typing import Optional

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging

import config as config_lib
import utils
from module import Module


class TfdsInput(Module):

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("dataset_name", None, "The tensorflow dataset name.")
        cfg.define("split", None, "The dataset split.")
        cfg.define("global_batch_size", None, "The global batch size.")
        cfg.define("is_training", None, "Whether the examples are used for training.")
        cfg.define("data_dir", None, "Used for tfds.load. If None, use $HOME/tensorflow_datasets.")
        cfg.define("download", False,
                   "Whether to download the examples. If false, use the local data under data_dir.")
        cfg.define(
            "shuffle_buffer_size",
            None,
            "The shuffle buffer size (only used when is_training=True).",
        )
        cfg.define("shuffle_seed", None, "The shuffle seed.")
        cfg.define("prefetch_buffer_size", tf.data.AUTOTUNE, "The prefetch buffer size.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.is_training is None:
            raise ValueError(f"{self.path()}: is_training must be specified explicitly")
        # TODO(ruoming): tune according to https://www.tensorflow.org/datasets/performances.
        builder = tfds.builder(cfg.dataset_name, data_dir=cfg.data_dir)
        if cfg.download:
            logging.info("Downloading %s", cfg.dataset_name)
            builder.download_and_prepare()
            logging.info("Downloading %s done", cfg.dataset_name)
        if not cfg.is_training:
            split_ds = builder.as_dataset(split=cfg.split, batch_size=1)
            num_examples = len(split_ds)
            if num_examples % cfg.global_batch_size != 0:
                raise ValueError(f"Evaluation dataset size ({num_examples}) must be divisible by "
                                 f"global_batch_size ({cfg.global_batch_size}")
        split = cfg.split
        if cfg.global_batch_size % jax.process_count() != 0:
            raise ValueError(f"global_batch_size ({cfg.global_batch_size} must be divisible by "
                             f"process_count ({jax.process_count()})")
        batch_size = cfg.global_batch_size // jax.process_count()
        split = tfds.even_splits(split, n=jax.process_count(), drop_remainder=cfg.is_training)[jax.process_index()]
        ds: tf.data.Dataset = builder.as_dataset(
            split=split,
            shuffle_files=cfg.is_training,
        )
        ds = ds.map(self._process_example, num_parallel_calls=tf.data.AUTOTUNE)
        if cfg.is_training:
            ds = ds.repeat()
            ds = ds.shuffle(
                cfg.shuffle_buffer_size,
                seed=cfg.shuffle_seed,
                reshuffle_each_iteration=True,
            )
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(cfg.prefetch_buffer_size)
        self._dataset = ds

    def _process_example(self, example):
        """Can be overridden by subclassses."""
        return example

    def __iter__(self):
        for example in self._dataset:
            yield utils.as_tensor(example)

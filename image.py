"""
References:
- https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py
"""
from typing import Optional
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging

import config as config_lib
from module import Module
import utils


MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


class ImagenetInput(Module):

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("tfds_name", "imagenet2012", "The tensorflow dataset name.")
        cfg.define("split", "validation", "The dataset split.")
        cfg.define("batch_size", None, "The batch size.")
        cfg.define("is_training", None, "Whether the examples are used for training.")
        cfg.define("download", False,
                   "Whether to download the examples. If false, use the local data under $HOME/tensorflow_datasets.")
        cfg.define(
            "shuffle_buffer_size",
            4096,
            "The shuffle buffer size (only used when is_training=True).",
        )
        cfg.define("shuffle_seed", None, "The shuffle seed.")
        cfg.define("prefetch_buffer_size", 1024, "The prefetch buffer size.")
        cfg.define("image_size", (224, 224), "The image size.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.is_training is None:
            raise ValueError(f"{self.path()}: is_training must be specified explicitly")
        ds: tf.data.Dataset = tfds.load(
            name=cfg.tfds_name,
            split=cfg.split,
            shuffle_files=cfg.is_training,
            download=cfg.download,
        )
        ds = ds.map(
            self._process_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        if cfg.is_training:
            ds = ds.repeat()
            ds = ds.shuffle(
                cfg.shuffle_buffer_size,
                seed=cfg.shuffle_seed,
                reshuffle_each_iteration=True,
            )
        ds = ds.prefetch(cfg.prefetch_buffer_size)
        ds = ds.batch(cfg.batch_size, drop_remainder=True)
        self._dataset = ds

    def _process_example(self, example):
        cfg = self.config
        logging.debug("example=%s", example.keys())
        image = example["image"]
        logging.info("image=%s", type(image))
        image = tf.cast(tf.convert_to_tensor(image), tf.float32)
        image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
        image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
        if cfg.is_training:
            image = tf.image.random_flip_left_right(image)
        image = tf.image.resize([image], cfg.image_size, method=tf.image.ResizeMethod.BICUBIC)[0]
        return {"image": image, "label": example["label"]}

    def __iter__(self):
        for example in self._dataset:
            yield utils.as_tensor(example)

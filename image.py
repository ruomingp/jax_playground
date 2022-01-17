"""
References:
- https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py
"""
import tensorflow as tf
from absl import logging

from input_tfds import TfdsInput

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


class ImagenetInput(TfdsInput):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("image_size", (224, 224), "The image size.")
        cfg.dataset_name = "imagenet2012"
        cfg.shuffle_buffer_size = 8192  # to be tuned.
        return cfg

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
        image = tf.image.resize(
            [image], cfg.image_size, method=tf.image.ResizeMethod.BICUBIC
        )[0]
        return {"image": image, "label": example["label"]}

"""Image input modules.

References:
- https://github.com/pytorch/vision/blob/main/torchvision/prototype/models/resnet.py
- https://github.com/pytorch/vision/blob/main/references/classification/presets.py
- https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py
- https://github.com/google-research/vision_transformer/blob/main/vit_jax/input_pipeline.py#L195-L241
"""
import jax
import numpy as np
import tensorflow as tf
from absl import logging

import config as config_lib
import utils
from input_tfds import TfdsInput
from module import Module

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def _whiten(image: np.ndarray) -> tf.Tensor:
    image = tf.cast(tf.convert_to_tensor(image), tf.float32)
    image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
    return image


def _random_crop(
    image: tf.Tensor, aspect_ratio_range=(0.75, 1.33), area_range=(0.08, 1.0), max_attempts=100
):
    """Generates a randomly cropped image.

    Args:
      image: `Tensor` of shape [H, W, C].
      aspect_ratio_range: An optional list of `float`s. The cropped area of the
          image must have an aspect ratio = width / height within this range.
      area_range: An optional list of `float`s. The cropped area of the image
          must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
          region of the image of the specified constraints. After `max_attempts`
          failures, return the entire image.
    Returns:
      Cropped image `Tensor`, [H', W', C].
    """
    # A bounding box covering the entire image.
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    # See `tf.image.sample_distorted_bounding_box` for more documentation.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.0,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True,
    )
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box
    # Crop the image to the specified bounding box.
    return tf.slice(image, bbox_begin, bbox_size)


def _central_crop(image, image_size):
    image_height, image_width, image_channels = image.shape
    offset_height = ((image_height - image_size[0]) + 1) // 2
    offset_width = ((image_width - image_size[1]) + 1) // 2
    return tf.slice(
        image, [offset_height, offset_width, 0], [image_size[0], image_size[1], image_channels]
    )


class ImagenetInput(TfdsInput):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("image_size", (224, 224), "The image size.")
        cfg.define(
            "eval_resize", (256, 256), "The image size to resize to during eval before cropping."
        )
        cfg.dataset_name = "imagenet2012"
        cfg.shuffle_buffer_size = 1024  # to be tuned.
        return cfg

    def _process_example(self, example):
        cfg = self.config
        logging.debug("example=%s", example.keys())
        image = example["image"]
        logging.info("image=%s", type(image))
        image = _whiten(image)
        if cfg.is_training:
            image = _random_crop(image)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.resize([image], cfg.image_size, method=tf.image.ResizeMethod.BILINEAR)[
                0
            ]
        else:
            image = tf.image.resize(
                [image], cfg.eval_resize, method=tf.image.ResizeMethod.BILINEAR
            )[0]
            image = _central_crop(image, cfg.image_size)
        return {"image": image, "label": example["label"]}


class FakeImagenetInput(Module):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("global_batch_size", 0, "The global batch size.")
        cfg.define(
            "total_num_batches",
            None,
            "The total number of batches. If None, unlimited.",
        )
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent=None):
        super().__init__(cfg, parent=parent)
        self._prng_key = jax.random.PRNGKey(1)
        self._num_batches = 0

    def __iter__(self):
        self._num_batches = 0
        return self

    def __next__(self):
        cfg = self.config
        self._num_batches += 1
        if cfg.total_num_batches is not None and self._num_batches > cfg.total_num_batches:
            raise StopIteration()
        self._prng_key, image_key, label_key = jax.random.split(self._prng_key, 3)
        if cfg.global_batch_size <= 0 or cfg.global_batch_size % jax.process_count() != 0:
            raise ValueError(
                f"Global batch size ({cfg.global_batch_size}) "
                f"must be positive and divisible by process count ({jax.process_count()})"
            )
        batch_size = cfg.global_batch_size // jax.process_count()
        image = jax.random.randint(
            image_key,
            shape=[batch_size, 224, 224, 3],
            minval=0,
            maxval=256,
            dtype=np.int32,
        )
        image = _whiten(image)
        label = jax.random.randint(
            label_key,
            shape=[batch_size],
            minval=0,
            maxval=1000,
            dtype=np.int32,
        )
        return utils.as_tensor(dict(image=image, label=label))

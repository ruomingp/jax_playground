"""
References:
- https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py
"""
import os.path
from typing import Optional

import jax.random
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags, logging

import config as config_lib
import learner
import resnet
from module import Module
from trainer import SpmdTrainer, SpmdEvaler

flags.DEFINE_string(
    "dir", None,
    "The root directory of the trainer. "
    "Checkpoints will be stored in <dir>/checkpoints. "
    "Summaries will be stored in <dir>/summaries.",
    required=True)

FLAGS = flags.FLAGS

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


class ImagenetInput(Module):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("is_training", False, "Whether the examples are used for training.")
        cfg.define("split", "train", "The dataset split.")
        cfg.define("batch_size", None, "The batch size.")
        cfg.define(
            "shuffle_buffer_size",
            4096,
            "The shuffle buffer size (only used when is_training=True).",
        )
        cfg.define("shuffle_seed", None, "The shuffle seed.")
        cfg.define("prefetch_buffer_size", 1024, "The prefetch buffer size.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        ds: tf.data.Dataset = tfds.load(
            name="imagenet2012",
            split=cfg.split,
            shuffle_files=cfg.is_training,
            download=False,
        )
        ds = ds.map(
            self._process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
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

    def _process_image(self, image):
        cfg = self.config
        image = tf.cast(image, tf.float32)
        image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
        image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
        if cfg.is_training:
            image = tf.image.random_flip_left_right(image)
        return image

    def __iter__(self):
        return iter(self._dataset)


def imagenet_trainer_config():
    num_train_examples = 1_281_167
    train_batch_size = 256
    eval_batch_size = 80  # divides 50_000 and can be divided by number of devices (8)
    steps_per_epoch = num_train_examples // train_batch_size

    cfg = SpmdTrainer.default_config()
    cfg.name = "imagenet_trainer"
    cfg.input = ImagenetInput.default_config().set(
        split="train", is_training=True, batch_size=train_batch_size
    )
    cfg.model = resnet.ResNetModel.resnet18_config()
    cfg.summary_writer.dir = os.path.join(FLAGS.dir, "summaries")
    cfg.summary_writer.write_every_n_steps = 10
    cfg.checkpointer.dir = os.path.join(FLAGS.dir, "checkpoints")
    cfg.checkpointer.write_every_n_steps = steps_per_epoch
    cfg.checkpointer.keep_every_n_steps = steps_per_epoch * 10
    evaler_train = SpmdEvaler.default_config().set(
        name="evaler_train",
        input=ImagenetInput.default_config().set(
            split="train[0:50000]", is_training=False, batch_size=eval_batch_size
        ),
    )
    evaler_validation = SpmdEvaler.default_config().set(
        name="evaler_validation",
        input=ImagenetInput.default_config().set(
            split="validation", is_training=False, batch_size=eval_batch_size
        ),
    )
    cfg.evalers = (evaler_train, evaler_validation)

    def learning_rate_schedule(step: int) -> float:
        stage = step // (steps_per_epoch * 30)
        return 0.1 * (10 ** -stage)

    cfg.learner = learner.Learner.default_config().set(
        weight_decay=1e-4,
        optimizer=config_lib.InstantiableConfig.for_function(optax.sgd).set(
            learning_rate=learning_rate_schedule, momentum=0.9
        ),
    )
    return cfg


def main(argv):
    trainer_config = imagenet_trainer_config()
    logging.info("Trainer config: %s", trainer_config.debug_string())
    trainer: SpmdTrainer = trainer_config.instantiate(parent=None)
    prng_key = jax.random.PRNGKey(1)
    trainer.run(prng_key, max_step=100)


if __name__ == "__main__":
    app.run(main)

"""
References:
- https://github.com/google/flax/blob/main/examples/imagenet/input_pipeline.py
"""
import jax.random
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

import config
import learner
import resnet
import trainer


class ImagenetInput(config.Configurable):

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("is_training", False, "Whether the examples are used for training.")
        cfg.define("split", "train", "The dataset split.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._dataset: tf.data.Dataset = tfds.load(
            name='imagenet2012', split=cfg.split, shuffle_files=cfg.is_training, try_gcs=True)


class ImagenetEvaler(trainer.SpmdEvaler):

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("metrics", None, "The eval metric config.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config


def imagenet_trainer_config():
    cfg = trainer.SpmdTrainer.default_config()
    cfg.input = config.InstantiableConfig.for_function(tfds.load).set(name='imagenet2012', split='train',
                                                                      shuffle_files=True, try_gcs=True)
    cfg.model = resnet.ResNetModel.resnet18_config()
    evaler_train = trainer.SpmdEvaler.default_config().set(
        name='evaler_train',
        input=ImagenetInput.default_config().set(split='train[0:50000]', is_training=False),
        metrics=ImagenetMetrics.default_config(),
    )
    evaler_validation = trainer.SpmdEvaler.default_config().set(
        name='evaler_validation',
        input=ImagenetInput.default_config().set(split='validation', is_training=False))
    cfg.evalers = (evaler_train, evaler_validation)

    def learning_rate_schedule(step: int) -> float:
        steps_per_epoch = 5004
        stage = step // (steps_per_epoch * 30)
        return 0.1 * (10 ** -stage)

    cfg.learner = learner.Learner.default_config().set(
        weight_decay=1e-4,
        optimizer=config.InstantiableConfig.for_function(optax.sgd).set(
            learning_rate=learning_rate_schedule, momentum=0.9))
    return cfg


def main(argv):
    trainer: trainer.SpmdTrainer = imagenet_trainer_config().instantiate(parent=None)
    prng_key = jax.random.PRNGKey(1)
    num_training_steps = 100
    for step in range(num_training_steps):
        prng_key, step_key = jax.random.split(prng_key)
        trainer.run_step(step, step_key)

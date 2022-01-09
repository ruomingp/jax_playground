import jax.random
import numpy as np
import optax
import tensorflow_datasets as tfds

import config
import learner
import resnet
import trainer


class DummyInput:

    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self._prng_key = jax.random.PRNGKey(1)

    def __iter__(self):
        return self

    def __next__(self):
        self._prng_key, image_key, label_key = jax.random.split(self._prng_key, 3)
        return dict(image=jax.random.randint(image_key, shape=[self.batch_size, 224, 224, 3], minval=0, maxval=256,
                                             dtype=np.int32),
                    label=jax.random.randint(label_key, shape=[self.batch_size], minval=0, maxval=1000, dtype=np.int32))


def imagenet_trainer_config():
    cfg = trainer.SpmdTrainer.default_config()
    cfg.input = config.InstantiableConfig.for_function(tfds.load).set(name='imagenet2012', split='train',
                                                                      shuffle_files=True, try_gcs=True)
    cfg.model = resnet.ResNetModel.resnet18_config()

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

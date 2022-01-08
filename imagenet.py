import jax.random
import tensorflow_datasets as tfds

import optax
import config
import resnet
import learner
import trainer


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
        optimizer=config.InstantiableConfig.for_function(optax.sgd).set(
            learning_rate=learning_rate_schedule, momentum=0.9, weight_decay=1e-4))
    return cfg


def main(argv):
    trainer: trainer.SpmdTrainer = imagenet_trainer_config().instantiate(parent=None)
    prng_key = jax.random.PRNGKey(1)
    num_training_steps = 100
    for step in range(num_training_steps):
        prng_key, step_key = jax.random.split(prng_key)
        trainer.run_step(step, step_key)

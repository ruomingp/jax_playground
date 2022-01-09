import tempfile

import jax.random
from absl.testing import absltest
from absl import flags

import config as config_lib
import imagenet
import learner
import optax
import resnet
from trainer import SpmdTrainer

FLAGS = flags.FLAGS


class TrainerTest(absltest.TestCase):
    def testTrainer(self):
        cfg = SpmdTrainer.default_config().set(name="test_trainer")
        cfg.dir = tempfile.mkdtemp()
        cfg.model = resnet.ResNetModel.resnet18_config().set(hidden_dim=16)
        cfg.input = config_lib.InstantiableConfig.for_class(imagenet.DummyInput)
        cfg.learner = learner.Learner.default_config().set(
            optimizer=config_lib.InstantiableConfig.for_function(optax.sgd).set(
                learning_rate=0.1, momentum=0.9))
        trainer: SpmdTrainer = cfg.instantiate()

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        trainer.init(init_key)
        for step in range(10):
            prng_key, run_key = jax.random.split(prng_key)
            trainer.run_step(step=step, prng_key=run_key)


if __name__ == "__main__":
    absltest.main()

import os.path
import tempfile

import jax
import jax.random
from jax.experimental import maps
from jax.experimental import mesh_utils
from absl.testing import absltest, parameterized
from absl import flags, logging

import config as config_lib
import imagenet
import learner
import optax
import resnet
from trainer import SpmdTrainer

FLAGS = flags.FLAGS


class TrainerTest(parameterized.TestCase):

    @parameterized.parameters(
        ('cpu', (1, 1)),
        ('tpu', (8, 1)),
        ('tpu', (2, 4)),
    )
    def testTrainer(self, platform, mesh_shape):
        trainer_dir = tempfile.mkdtemp()
        cfg = SpmdTrainer.default_config().set(name="test_trainer")
        cfg.model = resnet.ResNetModel.resnet18_config().set(hidden_dim=16)
        cfg.input = config_lib.InstantiableConfig.for_class(imagenet.DummyInput)
        cfg.learner = learner.Learner.default_config().set(
            optimizer=config_lib.InstantiableConfig.for_function(optax.sgd).set(
                learning_rate=0.1, momentum=0.9))
        cfg.checkpointer.write_every_n_steps = 5
        cfg.checkpointer.dir = os.path.join(trainer_dir, "checkpoints")
        cfg.summary_writer.dir = os.path.join(trainer_dir, "summaries")
        run_trainer = lambda: self._runTrainer(cfg, num_steps=11)
        devices = jax.devices()
        if not all(device.platform == platform for device in devices):
            logging.info('Skipping test for %s on %s', platform, [device.platform for device in devices])
            return
        if platform == 'cpu':
            run_trainer()
        else:
            devices = mesh_utils.create_device_mesh(mesh_shape)
            mesh = maps.Mesh(devices, ("data", "model"))
            with maps.mesh(mesh.devices, mesh.axis_names):
                run_trainer()

    def _runTrainer(self, cfg, num_steps):
        trainer: SpmdTrainer = cfg.instantiate(parent=None)

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        trainer.init(init_key)
        for step in range(num_steps):
            prng_key, run_key = jax.random.split(prng_key)
            trainer.run_step(step=step, prng_key=run_key)


if __name__ == "__main__":
    absltest.main()

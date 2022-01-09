import tempfile

import jax
import jax.random
from jax.experimental import maps
from jax.experimental import mesh_utils
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
        cfg.checkpointer.write_every_n_steps = 5
        run_trainer = lambda: self._runTrainer(cfg, num_steps=11)
        devices = jax.devices()
        if len(devices) == 8 and all(device.platform == 'tpu' for device in devices):
            mesh_shape = (4, 2)
            devices = mesh_utils.create_device_mesh(mesh_shape)
            mesh = maps.Mesh(devices, ("data", "model"))
            with maps.mesh(mesh.devices, mesh.axis_names):
                run_trainer()
        else:
            run_trainer()

    def _runTrainer(self, cfg, num_steps):
        trainer: SpmdTrainer = cfg.instantiate()

        prng_key = jax.random.PRNGKey(123)
        prng_key, init_key = jax.random.split(prng_key)
        trainer.init(init_key)
        for step in range(num_steps):
            prng_key, run_key = jax.random.split(prng_key)
            trainer.run_step(step=step, prng_key=run_key)


if __name__ == "__main__":
    absltest.main()

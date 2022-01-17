import os.path
import tempfile
from typing import Optional

import jax
import jax.random
import numpy as np
import optax
from absl import flags, logging
from absl.testing import absltest, parameterized
from jax.experimental import maps
from jax.experimental import mesh_utils

import config as config_lib
import learner
import resnet
from module import Module
from trainer import SpmdTrainer, SpmdEvaler

FLAGS = flags.FLAGS


class DummyInput(Module):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("batch_size", 8, "The batch size.")
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
        if (
            cfg.total_num_batches is not None
            and self._num_batches > cfg.total_num_batches
        ):
            raise StopIteration()
        self._prng_key, image_key, label_key = jax.random.split(self._prng_key, 3)
        return dict(
            image=jax.random.randint(
                image_key,
                shape=[cfg.batch_size, 224, 224, 3],
                minval=0,
                maxval=256,
                dtype=np.int32,
            ),
            label=jax.random.randint(
                label_key, shape=[cfg.batch_size], minval=0, maxval=1000, dtype=np.int32
            ),
        )


class TrainerTest(parameterized.TestCase):
    @parameterized.parameters(
        ("cpu", (1, 1)),
        ("tpu", (8, 1)),
        ("tpu", (2, 4)),
    )
    def testTrainer(self, platform, mesh_shape):
        trainer_dir = tempfile.mkdtemp()
        cfg = SpmdTrainer.default_config().set(name="test_trainer")
        cfg.model = resnet.ResNetModel.resnet18_config().set(
            hidden_dim=16, num_blocks_per_stage=[1]
        )
        cfg.input = DummyInput.default_config()
        cfg.learner = learner.Learner.default_config().set(
            optimizer=config_lib.config_for_function(optax.sgd).set(
                learning_rate=0.1, momentum=0.9
            )
        )
        evaler_cfg = SpmdEvaler.default_config().set(
            name="eval_dummy",
            input=DummyInput.default_config().set(total_num_batches=2),
        )
        evaler_cfg.summary_writer.dir = os.path.join(
            trainer_dir, "summaries", evaler_cfg.name
        )
        cfg.evalers = [evaler_cfg]
        cfg.checkpointer.write_every_n_steps = 5
        cfg.checkpointer.dir = os.path.join(trainer_dir, "checkpoints")
        cfg.summary_writer.dir = os.path.join(trainer_dir, "summaries", "train")
        run_trainer = lambda: self._runTrainer(cfg, num_steps=11)
        devices = jax.devices()
        if not all(device.platform == platform for device in devices):
            logging.info(
                "Skipping test for %s on %s",
                platform,
                [device.platform for device in devices],
            )
            return
        if platform == "cpu":
            run_trainer()
        else:
            devices = mesh_utils.create_device_mesh(mesh_shape)
            mesh = maps.Mesh(devices, ("data", "model"))
            with maps.mesh(mesh.devices, mesh.axis_names):
                run_trainer()

    def _runTrainer(self, cfg, num_steps):
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        trainer.run(prng_key=jax.random.PRNGKey(123), max_step=10)


if __name__ == "__main__":
    absltest.main()

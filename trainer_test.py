import tempfile
from typing import Optional

import jax
import jax.random
import numpy as np
from absl import flags, logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp

import config as config_lib
import layers
import learner
import optimizers
import param_init
from checkpointer import Checkpointer
from module import BaseLayer, Module, Tensor
from trainer import SpmdEvaler, SpmdTrainer

FLAGS = flags.FLAGS

NUM_CLASSES = 16


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
        if cfg.total_num_batches is not None and self._num_batches > cfg.total_num_batches:
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
                label_key,
                shape=[cfg.batch_size],
                minval=0,
                maxval=NUM_CLASSES,
                dtype=np.int32,
            ),
        )


class DummyModel(BaseLayer):
    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "fc",
            layers.Linear.default_config().set(
                input_dim=3,
                output_dim=NUM_CLASSES,
                bias=True,
                param_partition_spec=(None, "model"),
            ),
        )

    def forward(self, image: Tensor, label: Tensor):
        # [batch, 3].
        hidden = image.mean(axis=(1, 2))
        logits: Tensor = self.fc(hidden)
        loss = (
            -(jax.nn.log_softmax(logits) * jax.nn.one_hot(label, NUM_CLASSES, dtype=logits.dtype))
            .sum(axis=-1)
            .mean()
        )
        return loss, {"prng_key": self.prng_key}


class TrainerTest(parameterized.TestCase):
    @parameterized.parameters(
        ("cpu", None),
        ("cpu", (1, 1)),
        ("tpu", (8, 1)),
        ("tpu", (2, 4)),
    )
    def testTrainer(self, platform, mesh_shape):
        devices = jax.devices()
        if not all(device.platform == platform for device in devices):
            logging.info(
                "Skipping test for %s on %s",
                platform,
                [device.platform for device in devices],
            )
            return
        cfg = SpmdTrainer.default_config().set(name="test_trainer")
        cfg.dir = tempfile.mkdtemp()
        if mesh_shape is not None:
            cfg.mesh_axis_names = ("data", "model")
        else:
            cfg.mesh_axis_names = ("data",)
        cfg.mesh_shape = mesh_shape
        cfg.model = DummyModel.default_config().set(
            dtype=jnp.float32, param_init=param_init.DefaultInitializer.default_config()
        )
        cfg.input = DummyInput.default_config()
        cfg.learner = learner.Learner.default_config().set(
            optimizer=config_lib.config_for_function(optimizers.sgd_optimizer).set(
                learning_rate=0.1,
                momentum=0.9,
                weight_decay=1e-4,
            )
        )
        cfg.evalers = dict(
            eval_dummy=SpmdEvaler.default_config().set(
                input=DummyInput.default_config().set(total_num_batches=2),
            )
        )
        cfg.checkpointer.write_every_n_steps = 5
        cfg.max_step = 12
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        output_a = trainer.run(prng_key=jax.random.PRNGKey(123))

        ckpt: Checkpointer = (
            Checkpointer.default_config()
            .set(name="ckpt", dir=trainer.checkpointer.config.dir)
            .instantiate(parent=None)
        )
        restored_step, restored_state = ckpt.restore(step=None, state=trainer._trainer_state)
        self.assertEqual(10, restored_step)
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        # Since we will be resuming from the checkpoint at step 10, a different prng_key doesn't matter.
        output_b = trainer.run(prng_key=jax.random.PRNGKey(456))
        # The prng_key per step is deterministic.
        np.testing.assert_array_equal(output_a["aux"]["prng_key"], output_b["aux"]["prng_key"])


if __name__ == "__main__":
    absltest.main()

"""A launcher to train ResNet-18 on ImageNet.

On the TPU VM:
gs_bucket=permanent-us-central1-q5loch
dir=gs://${gs_bucket}/${USER}/experiments/imagenet-dummy-inputs
echo $dir
python3 imagenet_with_dummy_inputs.py --dir=$dir --interval=1000000 2>&1 | tee /tmp/log

On your local machine:
pip install tensorflow tbp-nightly
gcloud auth application-default login
tensorboard --logdir=$dir/summaries
"""
import os.path

from absl import app, flags, logging
import jax  # jax must be imported before tensorflow!
from jax.experimental import maps
from jax.experimental import mesh_utils
import numpy as np

import config as config_lib
import learner
import resnet
import schedule
from module import Module
from trainer import SpmdTrainer, SpmdEvaler

flags.DEFINE_string(
    "dir",
    None,
    "The root directory of the trainer. "
    "Checkpoints will be stored in <dir>/checkpoints. "
    "Summaries will be stored in <dir>/summaries.",
    required=True,
)
flags.DEFINE_list(
    "mesh_shape", [8, 1], "The global device mesh shape for (data, model)."
)
flags.DEFINE_integer("jax_profiler_port", None, "If not None, the profiler port.")
flags.DEFINE_integer("interval", None, "If not None, the number of steps between ckpt and eval.")

FLAGS = flags.FLAGS


class DummyInput(Module):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("global_batch_size", 256, "The batch size.")
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
                shape=[cfg.global_batch_size, 224, 224, 3],
                minval=0,
                maxval=256,
                dtype=np.int32,
            ),
            label=jax.random.randint(
                label_key, shape=[cfg.global_batch_size], minval=0, maxval=1000, dtype=np.int32
            ),
        )


def imagenet_trainer_config():
    num_train_examples = 1_281_167
    train_batch_size = 256
    eval_batch_size = 80  # divides 50_000 and can be divided by number of devices (8)
    steps_per_epoch = num_train_examples // train_batch_size

    cfg = SpmdTrainer.default_config()
    cfg.name = "imagenet_trainer"

    # Model and optimization.
    cfg.model = resnet.ResNetModel.resnet18_config()
    learning_rate = config_lib.config_for_function(schedule.stepwise).set(
        sub=[0.1, 0.01, 0.001],
        start_step=[steps_per_epoch * 30, steps_per_epoch * 60],
    )
    cfg.learner = learner.Learner.default_config().set(
        optimizer=config_lib.config_for_function(learner.sgd_optimizer).set(
            learning_rate=learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        ),
    )

    # Training inputs.
    cfg.input = DummyInput.default_config().set(
        global_batch_size=train_batch_size,
    )

    # Evaluation.
    evaler_train = SpmdEvaler.default_config().set(
        name="eval_train",
        input=DummyInput.default_config().set(
            total_num_batches=160 if FLAGS.interval else 50000
        ),
    )
    # cfg.evalers = (evaler_train,)
    cfg.evalers = []

    # Summaries and checkpoints.
    cfg.checkpointer.dir = os.path.join(FLAGS.dir, "checkpoints")
    cfg.checkpointer.write_every_n_steps = FLAGS.interval or steps_per_epoch
    cfg.checkpointer.keep_every_n_steps = cfg.checkpointer.write_every_n_steps * 10
    summary_dir = os.path.join(FLAGS.dir, "summaries")
    cfg.summary_writer.write_every_n_steps = FLAGS.interval
    cfg.summary_writer.dir = os.path.join(summary_dir, "train_train")
    cfg.vlog = 0  # Set to 5 to enable verbose logging.
    for evaler_cfg in cfg.evalers:
        evaler_cfg.vlog = 0
        evaler_cfg.run_every_n_steps = FLAGS.interval or steps_per_epoch
        evaler_cfg.input.set(global_batch_size=eval_batch_size)
        evaler_cfg.summary_writer.dir = os.path.join(summary_dir, evaler_cfg.name)
    return cfg


def run_trainer(trainer_config, mesh_shape):
    trainer: SpmdTrainer = trainer_config.instantiate(parent=None)
    prng_key = jax.random.PRNGKey(1)
    run = lambda: trainer.run(prng_key, max_step=(FLAGS.interval * 10 if FLAGS.interval else 5004 * 90))
    if mesh_shape:
        devices = mesh_utils.create_device_mesh(mesh_shape)
        mesh = maps.Mesh(devices, ("data", "model"))
        with maps.mesh(mesh.devices, mesh.axis_names):
            run()
    else:
        run()


def main(argv):
    # Start jax.profiler for Tensorboard and profiling in open source.
    if FLAGS.jax_profiler_port is not None:
        server = jax.profiler.start_server(FLAGS.jax_profiler_port)

    logging.info("Creating trainer config")
    trainer_config = imagenet_trainer_config()
    logging.info("Trainer config: %s", trainer_config.debug_string())
    run_trainer(trainer_config, FLAGS.mesh_shape)


if __name__ == "__main__":
    app.run(main)

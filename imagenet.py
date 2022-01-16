"""
On the TPU VM:
dir=gs://permanent-us-central1-q5loch/${USER}/experiments/imagenet-$(date +%F)b
echo $dir
python3 imagenet.py --dir=$dir --data_dir=gs://permanent-us-central1-q5loch/tensorflow_datasets 2>&1 | tee /tmp/log

On your local machine:
pip install tensorflow tbp-nightly
gcloud auth application-default login
tensorboard --logdir=$dir/summaries
"""
import os.path

import jax
import optax
from absl import app, flags, logging
from jax.experimental import maps
from jax.experimental import mesh_utils

import config as config_lib
import learner
import resnet
from image import ImagenetInput
from trainer import SpmdTrainer, SpmdEvaler

flags.DEFINE_string(
    "dir", None,
    "The root directory of the trainer. "
    "Checkpoints will be stored in <dir>/checkpoints. "
    "Summaries will be stored in <dir>/summaries.",
    required=True)
flags.DEFINE_string("data_dir", None, "The tfds directory. If None, uses ~/tensorflow_datasets.")
flags.DEFINE_list("mesh_shape", [8, 1], "The global device mesh shape for (data, model).")
flags.DEFINE_integer("jax_profiler_port", 9999, "The profiler port.")


FLAGS = flags.FLAGS


def imagenet_trainer_config():
    num_train_examples = 1_281_167
    train_batch_size = 256
    eval_batch_size = 80  # divides 50_000 and can be divided by number of devices (8)
    steps_per_epoch = num_train_examples // train_batch_size

    cfg = SpmdTrainer.default_config()
    cfg.name = "imagenet_trainer"
    read_parallelism, process_parallelism = 64, 1024
    cfg = ImagenetInput.default_config().set(
        split="train", is_training=True, global_batch_size=train_batch_size,
        data_dir=FLAGS.data_dir,
        read_parallelism=read_parallelism,
        process_parallelism=process_parallelism,
        prefetch_buffer_size=read_parallelism * 1024,
        shuffle_buffer_size=read_parallelism * 1024)

    cfg.model = resnet.ResNetModel.resnet18_config()
    cfg.summary_writer.write_every_n_steps = 100
    cfg.checkpointer.dir = os.path.join(FLAGS.dir, "checkpoints")
    cfg.checkpointer.write_every_n_steps = steps_per_epoch
    cfg.checkpointer.keep_every_n_steps = steps_per_epoch * 10
    evaler_train = SpmdEvaler.default_config().set(
        name="eval_train",
        input=ImagenetInput.default_config().set(split="train[0:50000]"),
    )
    evaler_validation = SpmdEvaler.default_config().set(
        name="eval_validation",
        input=ImagenetInput.default_config().set(split="validation"),
    )
    cfg.evalers = (evaler_train, evaler_validation)

    summary_dir = os.path.join(FLAGS.dir, "summaries")
    cfg.summary_writer.dir = os.path.join(summary_dir, "train_train")
    for evaler_cfg in cfg.evalers:
        evaler_cfg.run_every_n_steps = steps_per_epoch
        evaler_cfg.input.set(is_training=False, global_batch_size=eval_batch_size, data_dir=FLAGS.data_dir)
        evaler_cfg.summary_writer.dir = os.path.join(summary_dir, evaler_cfg.name)

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


def run_trainer(trainer_config, mesh_shape):
    trainer: SpmdTrainer = trainer_config.instantiate(parent=None)
    prng_key = jax.random.PRNGKey(1)
    devices = mesh_utils.create_device_mesh(mesh_shape)
    mesh = maps.Mesh(devices, ("data", "model"))
    with maps.mesh(mesh.devices, mesh.axis_names):
        trainer.run(prng_key, max_step=1000000)


def main(argv):
    # Start jax.profiler for Tensorboard and profiling in open source.
    if FLAGS.jax_profiler_port is not None:
        server = jax.profiler.start_server(FLAGS.jax_profiler_port)

    trainer_config = imagenet_trainer_config()
    logging.info("Trainer config: %s", trainer_config.debug_string())
    run_trainer(trainer_config, FLAGS.mesh_shape)


if __name__ == "__main__":
    app.run(main)

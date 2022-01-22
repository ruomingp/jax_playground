"""A launcher to train ResNet-18 on ImageNet.

On the TPU VM:
gs_bucket=permanent-us-central1-q5loch
dir=gs://${gs_bucket}/${USER}/experiments/imagenet-$(date +%F)a
echo $dir
data_dir=gs://${gs_bucket}/tensorflow_datasets
echo $data_dir
python3 imagenet.py --dir=$dir --data_dir=$data_dir 2>&1 | tee /tmp/log

On your local machine:
pip install tensorflow tbp-nightly
gcloud auth application-default login
tensorboard --logdir=$dir/summaries
"""
import os.path

import tensorflow as tf
import jax
from absl import app, flags, logging
from jax.experimental import maps
from jax.experimental import mesh_utils

import config as config_lib
import learner
import resnet
import schedule
from image import ImagenetInput
from trainer import SpmdTrainer, SpmdEvaler

flags.DEFINE_string(
    "dir",
    None,
    "The root directory of the trainer. "
    "Checkpoints will be stored in <dir>/checkpoints. "
    "Summaries will be stored in <dir>/summaries.",
    required=True,
)
flags.DEFINE_string(
    "data_dir",
    None,
    "The tfds directory. If None, uses ~/tensorflow_datasets.",
)
flags.DEFINE_list(
    "mesh_shape", [8, 1], "The global device mesh shape for (data, model)."
)
flags.DEFINE_integer("jax_profiler_port", None, "If not None, the profiler port.")
flags.DEFINE_bool("debug", False, "If true, run in the debug mode.")

FLAGS = flags.FLAGS


def imagenet_trainer_input_config():
    train_batch_size = 256

    # Training inputs.
    read_parallelism = 1
    cfg = ImagenetInput.default_config().set(
        name="imagenet_train_input",
        split="train",
        is_training=True,
        global_batch_size=train_batch_size,
        data_dir=FLAGS.data_dir,
        read_parallelism=read_parallelism,
        decode_parallelism=128,
        process_parallelism=1024,
        prefetch_buffer_size=64 * 1024,
        shuffle_buffer_size=read_parallelism * 1024,
    )
    return cfg


def main(argv):
    # Start jax.profiler for Tensorboard and profiling in open source.
    if FLAGS.jax_profiler_port is not None:
        server = jax.profiler.start_server(FLAGS.jax_profiler_port)

    trainer_input_config = imagenet_trainer_input_config()
    logging.info("Trainer config: %s", trainer_input_config.debug_string())
    trainer_inputs = trainer_input_config.instantiate(parent=None)
    # run_trainer(trainer_config, FLAGS.mesh_shape)


if __name__ == "__main__":
    # tf.compat.v1.app.run(main)
    app.run(main)

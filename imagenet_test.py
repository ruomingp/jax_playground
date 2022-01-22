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
import tensorflow_datasets as tfds
import jax
from absl import app, flags, logging
from jax.experimental import maps
from jax.experimental import mesh_utils

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


MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def _process_example(example):
    image = example["image"]
    image = tf.cast(tf.convert_to_tensor(image), tf.float32)
    image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image = tf.image.resize([image], (224, 224), method=tf.image.ResizeMethod.BICUBIC)[0]
    image = tf.image.random_flip_left_right(image)
    return {"image": image, "label": example["label"]}


def main(argv):
    batch_size=256
    builder = tfds.builder("imagenet2012", data_dir="gs://permanent-us-central1-q5loch/tensorflow_datasets")
    split = tfds.even_splits("train", n=jax.process_count(), drop_remainder=True)[jax.process_index()]
    read_config = tfds.ReadConfig(interleave_cycle_length=1, num_parallel_calls_for_interleave_files=1, num_parallel_calls_for_decode=128)
    ds: tf.data.Dataset = builder.as_dataset(split=split, shuffle_files=True, read_config=read_config)
    ds = ds.map(_process_example, num_parallel_calls=32)
    ds = ds.shuffle(8192, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat()
    ds = ds.prefetch(8192)
    print(tfds.benchmark(ds, batch_size=batch_size, num_iter=100).stats)


if __name__ == "__main__":
    app.run(main)

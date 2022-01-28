"""A test program to reproduce a crash with tfds as_dataset().

On the TPU VM:
python3 imagenet_test.py 2>&1 | tee /tmp/log
"""
from absl import app, logging
import tensorflow as tf
import tensorflow_datasets as tfds
import guppy

import utils

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def _process_example(example):
    image = example["image"]
    image = tf.cast(tf.convert_to_tensor(image), tf.float32)
    image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image = tf.image.resize([image], (224, 224), method=tf.image.ResizeMethod.BICUBIC)[
        0
    ]
    image = tf.image.random_flip_left_right(image)
    return {"image": image, "label": example["label"]}


def main(argv):
    batch_size = 256
    builder = tfds.builder(
        "imagenet2012", data_dir="gs://permanent-us-central1-q5loch/tensorflow_datasets"
    )
    split = "train"
    read_config = tfds.ReadConfig(
        interleave_cycle_length=1,
        num_parallel_calls_for_interleave_files=1,
        num_parallel_calls_for_decode=4,
    )
    ds: tf.data.Dataset = builder.as_dataset(
        split=split, shuffle_files=True, read_config=read_config
    )
    ds = ds.map(_process_example, num_parallel_calls=64)
    ds = ds.shuffle(1024, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat()
    ds = ds.prefetch(4)
    for i, batch in enumerate(ds):
        logging.info(f"Batch {i}: {utils.shapes(batch)}")
        if i % 100 == 101:
            h = guppy.hpy()
            logging.info("Heapy: %s", h.heap())


if __name__ == "__main__":
    app.run(main)

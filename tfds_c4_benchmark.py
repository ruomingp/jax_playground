import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, logging

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def _process_example(example):
    logging.log_first_n(logging.INFO, "example=%s", 1, example.keys())
    return example


def main(argv):
    batch_size = 128
    builder = tfds.builder(
        "c4",
        # data_dir="gs://permanent-us-central1-q5loch/tensorflow_datasets",
        data_dir="gs://permanent-us-east1-q5loch/tensorflow_datasets"
    )
    split = tfds.even_splits("train", n=jax.process_count(), drop_remainder=True)[
        jax.process_index()
    ]
    read_config = tfds.ReadConfig(
        interleave_cycle_length=1,
        num_parallel_calls_for_interleave_files=1,
        num_parallel_calls_for_decode=16,
    )
    ds: tf.data.Dataset = builder.as_dataset(
        split=split, shuffle_files=True, read_config=read_config
    )
    ds = ds.map(_process_example, num_parallel_calls=32)
    ds = ds.shuffle(8192, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat()
    ds = ds.prefetch(8192)
    print(tfds.benchmark(ds, batch_size=batch_size, num_iter=100).stats)


if __name__ == "__main__":
    app.run(main)

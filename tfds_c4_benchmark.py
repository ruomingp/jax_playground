from typing import Sequence
import jax
import numpy as np
import seqio
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app


def perplexity(targets: Sequence[str], scores: Sequence[int]):
    return {
        "perplexity": seqio.metrics.Scalar(np.exp(np.mean(scores)))
    }


# https://github.com/google-research/google-research/blob/77e1f14f3f7af7dc91dcdba7402ebc46c55ac2a6/primer/t5_tasks.py#L20
def add_task(*, tfds_data_dir, vocab_file):
    # https://github.com/google/seqio/blob/27c844b0ee2eea4a163ac6ad8cc56d280d673a11/seqio/dataset_providers.py#L1260.
    seqio.TaskRegistry.add(
        "lm_c4_en",
        # https://github.com/google/seqio/blob/27c844b0ee2eea4a163ac6ad8cc56d280d673a11/seqio/dataset_providers.py#L389.
        seqio.TfdsDataSource(tfds_name="c4/en:3.0.1", tfds_data_dir=tfds_data_dir),
        preprocessors=[
            lambda x: seqio.preprocessors.rekey(x, {"targets": "text"}), seqio.preprocessors.tokenize, seqio.preprocessors.append_eos
        ],
        output_features={
            "targets": seqio.Feature(
                seqio.SentencePieceVocabulary(vocab_file),
                add_eos=True, dtype=tf.int32
            ),
        },
        metric_fns=[perplexity])


def main(argv):
    batch_size = 128

    add_task(tfds_data_dir="gs://permanent-us-central1-q5loch/tensorflow_datasets",
             vocab_file="gs://permanent-us-central1-q5loch/tokenizers/sentencepiece/t5-base")

    dataset = seqio.get_mixture_or_task("lm_c4_en").get_dataset(
        sequence_length={"targets": 16},
        split="train",
        shuffle=True,
        num_epochs=1,
        shard_info=seqio.ShardInfo(index=jax.process_index(), num_shards=jax.process_count()),
        # https://github.com/google/seqio/blob/5f4213d73cad4b8d1507cb82191c1f310675725b/README.md#optional-offline-caching.
        use_cached=False,
        seed=42 + jax.process_index(),
    )
    for _, ex in zip(range(10), dataset.as_numpy_iterator()):
        print(ex)
    # print(tfds.benchmark(dataset, batch_size=batch_size, num_iter=100).stats)


if __name__ == "__main__":
    app.run(main)

import numpy as np
import pytest
import seqio
import tensorflow as tf
from absl.testing import absltest, parameterized

import config
import utils
from input_text import LmInput


def _count_batches(dataset, max_batches=100):
    for n, _ in enumerate(dataset):
        if n >= max_batches:
            return -1
    return n + 1


class LmInputWithFakeTextData(LmInput):
    def __init__(self, cfg: config.Config, parent, ds_fn):
        self._ds_fn = ds_fn
        super().__init__(cfg, parent=parent)

    def _seqio_data_source(self) -> seqio.DataSource:
        cfg = self.config
        return seqio.FunctionDataSource(self._ds_fn, splits=[cfg.split])


class LmInputTest(parameterized.TestCase):
    @parameterized.parameters(False, True)
    @pytest.mark.gs_login  # must annotate within @parameterized.parameters
    def testRealData(self, is_training):
        batch_size = 3
        cfg = LmInput.default_config().set(
            name="train" if is_training else "eval",
            dataset_name="c4/en:3.0.1",
            split=("train" if is_training else "validation") + f"[:{batch_size * 10}]",
            global_batch_size=batch_size,
            is_training=is_training,
            data_dir="gs://permanent-us-central1-q5loch/tensorflow_datasets",
            sentence_piece_vocab_file="gs://permanent-us-central1-q5loch/tokenizers/sentencepiece/t5-base",
            max_length=16,
        )
        dataset = cfg.instantiate(parent=None)
        if is_training:
            # For training, we loop over the dataset forever.
            self.assertEqual(-1, _count_batches(dataset, max_batches=24))
        else:
            # For evaluation, we loop over the dataset only once.
            self.assertEqual(10, _count_batches(dataset, max_batches=24))
        for batch in dataset:
            self.assertEqual(
                {key: (batch_size, cfg.max_length) for key in ("inputs", "targets")},
                utils.shapes(batch),
            )
            break

    @parameterized.parameters(False, True)
    @pytest.mark.gs_login  # must annotate within @parameterized.parameters
    def testFakeTextData(self, is_training):
        batch_size = 3
        cfg = LmInputWithFakeTextData.default_config().set(
            name="train" if is_training else "eval",
            dataset_name="c4/en:3.0.1",
            split="train" if is_training else "validation",
            global_batch_size=batch_size,
            is_training=is_training,
            data_dir="gs://permanent-us-central1-q5loch/tensorflow_datasets",
            sentence_piece_vocab_file="gs://permanent-us-central1-q5loch/tokenizers/sentencepiece/t5-base",
            max_length=16,
        )

        def data_gen():
            for _ in range(100):
                for index, text in enumerate(["hello world", "hello moon", "hello tiger"]):
                    yield {"text": text, "index": index}

        def ds_fn(split, shuffle_files):
            return tf.data.Dataset.from_generator(
                data_gen,
                output_signature={
                    "text": tf.TensorSpec(shape=(), dtype=tf.string),
                    "index": tf.TensorSpec(shape=(), dtype=tf.int32),
                },
            )

        dataset = cfg.instantiate(parent=None, ds_fn=ds_fn)
        for batch in dataset:
            self.assertEqual(
                {key: (batch_size, cfg.max_length) for key in ("inputs", "targets")},
                utils.shapes(batch),
            )
            if not is_training:
                # We shuffle data when is_training=True and cannot check batch contents.
                np.testing.assert_array_equal(
                    [
                        [21820, 296, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [21820, 8114, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [21820, 3, 17, 4424, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    batch["targets"],
                )
                np.testing.assert_array_equal(
                    [
                        [-1, 21820, 296, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-1, 21820, 8114, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [-1, 21820, 3, 17, 4424, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ],
                    batch["inputs"],
                )
            break


if __name__ == "__main__":
    absltest.main()

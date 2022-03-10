"""Text input modules.

References:
https://github.com/google-research/google-research/blob/77e1f14f3f7af7dc91dcdba7402ebc46c55ac2a6/primer/t5_tasks.py#L20
"""
import re
from typing import Sequence

import jax
import numpy as np
import seqio
import tensorflow as tf

import config as config_lib
import utils
from input_tfds import BaseTfdsInput
from module import Module


def perplexity(targets: Sequence[str], scores: Sequence[int]):
    return {"perplexity": seqio.metrics.Scalar(np.exp(np.mean(scores)))}


class LmInput(BaseTfdsInput):
    """Inputs for language models.

    Each input batch is a dict with "inputs" and "targets" as keys and int32 tensors of shape [batch_size, max_length]
    as values.
    """

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("sentence_piece_vocab_file", "", "The sentence piece vocab file.")
        cfg.define(
            "max_length",
            None,
            "The maximum sequence length, including the EOS token (when the sequence is not truncated).",
        )
        return cfg

    @property
    def _seqio_task_name(self):
        cfg = self.config
        return re.sub("[\[\]/]", "_", f"{self.path()}.{cfg.dataset_name}.{cfg.split}")

    def _seqio_data_source(self) -> seqio.DataSource:
        cfg = self.config
        return seqio.TfdsDataSource(tfds_name=cfg.dataset_name, tfds_data_dir=cfg.data_dir)

    def _build_dataset(self) -> tf.data.Dataset:
        cfg = self.config

        vocab = seqio.SentencePieceVocabulary(cfg.sentence_piece_vocab_file)

        seqio.TaskRegistry.add(
            self._seqio_task_name,
            self._seqio_data_source(),
            preprocessors=[
                lambda x: seqio.preprocessors.rekey(x, {"targets": "text"}),
                seqio.preprocessors.tokenize,
                seqio.preprocessors.append_eos,
            ],
            output_features={
                "targets": seqio.Feature(
                    vocab,
                    add_eos=True,
                    dtype=tf.int32,
                ),
            },
            metric_fns=[perplexity],
        )

        ds: tf.data.Dataset = seqio.get_mixture_or_task(self._seqio_task_name).get_dataset(
            sequence_length={"targets": cfg.max_length},
            split=cfg.split,
            shuffle=cfg.is_training,
            num_epochs=None if cfg.is_training else 1,
            shard_info=seqio.ShardInfo(index=jax.process_index(), num_shards=jax.process_count()),
            # TODO(ruoming): experiment with caching.
            # https://github.com/google/seqio/blob/master/README.md#optional-offline-caching.
            use_cached=False,
        )

        def add_inputs(example):
            targets = example["targets"]
            inputs = tf.concat([[vocab.tokenizer.bos_id()], targets[:-1]], axis=0)
            return {
                "targets": targets,
                "inputs": inputs,
            }

        ds = ds.map(add_inputs, num_parallel_calls=tf.data.AUTOTUNE)
        # TODO(ruoming): support trim_and_pack_dataset for training efficiency:
        # https://github.com/google/seqio/blob/772e714ba807eba3e89f599b609e39e872ef8dbc/seqio/utils.py#L261
        ds = seqio.utils.trim_and_pad_dataset(
            ds, {key: cfg.max_length for key in ("inputs", "targets")}
        )
        ds = ds.batch(self._per_process_batch_size(), drop_remainder=True)
        return ds


class FakeLmInput(Module):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("global_batch_size", 0, "The global batch size.")
        cfg.define(
            "total_num_batches",
            None,
            "The total number of batches. If None, unlimited.",
        )
        cfg.define("source_length", 1024, "The length of a sequence (in tokens).")
        cfg.define("max_token_id", 2048, "The maximum value a token-ID can take.")
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
        self._prng_key, tokens_key = jax.random.split(self._prng_key, 2)
        if cfg.global_batch_size <= 0 or cfg.global_batch_size % jax.process_count() != 0:
            raise ValueError(
                f"Global batch size ({cfg.global_batch_size}) "
                f"must be positive and divisible by process count ({jax.process_count()})"
            )
        batch_size = cfg.global_batch_size // jax.process_count()
        tokens = jax.random.randint(
            tokens_key,
            shape=[batch_size, cfg.source_length + 1],
            minval=0,
            maxval=cfg.max_token_id,
            dtype=np.int32,
        )
        return utils.as_tensor(
            dict(
                inputs=tokens[:, :-1],
                targets=tokens[:, 1:],
            )
        )

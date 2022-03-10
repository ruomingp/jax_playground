"""A trainer for causal LMs on fake data or c4/en."""
import logging
from typing import Optional

from absl import flags

import config as config_lib
from input_text import FakeLmInput, LmInput
from trainer import SpmdEvaler, SpmdTrainer

flags.DEFINE_string(
    "data_dir",
    "gs://permanent-us-central2-0rxn/tensorflow_datasets",
    "The tfds directory. If 'FAKE', uses input_text.FakeLmInput.",
)
flags.DEFINE_integer(
    "max_train_examples", None, "If not None, the maximum number of training examples per epoch."
)
flags.DEFINE_integer(
    "max_eval_examples",
    None,
    "If not None, the maximum number of eval examples. "
    "If there are more examples in an eval dataset, use only the first N examples.",
)

FLAGS = flags.FLAGS


def base_trainer_config(
    train_batch_size: int, eval_batch_size: int, max_sequence_length: int
) -> config_lib.InstantiableConfig:
    """Returns a base c4/en trainer config.

    Args:
        train_batch_size: the global batch size for training.
        eval_batch_size: the global batch size for evaluation. Must divide all evaluation sets.
        max_sequence_length: maximum sequence length of an example.

    Returns:
        A SpmdTrainer config, containing:
        - Training input config to use up to --max_train_examples examples from the train split.
        - A summary writer to save training summaries every 100 steps.
        - Two evaluator configs to run every epoch and save summaries (up to --max_eval_examples):
          * eval_train: the first 50K samples from the train split.
          * eval_validation: the validation split
        - A checkpointer config to save a ckpt every 1k steps.
    """
    num_train_examples = 364_868_901
    max_token_id = 32_768
    sentence_piece_vocab_file = "gs://permanent-us-central2-0rxn/tokenizers/sentencepiece/t5-base"
    train_split = "train"
    if FLAGS.max_train_examples is not None:
        num_train_examples = min(FLAGS.max_train_examples, num_train_examples)
        train_split = f"{train_split}[:{num_train_examples}]"

    steps_per_snapshot = 1000  # How often to run evaluation and checkpointing.

    cfg = SpmdTrainer.default_config()
    cfg.name = "c4_en_trainer"

    # Training inputs.
    if FLAGS.data_dir == "FAKE":
        logging.warning("Using FAKE inputs!")
        cfg.input = FakeLmInput.default_config().set(
            global_batch_size=train_batch_size,
            source_length=max_sequence_length,
            max_token_id=max_token_id,
        )
    else:
        cfg.input = LmInput.default_config().set(
            dataset_name="c4/en:3.0.1",
            split="train",
            global_batch_size=train_batch_size,
            is_training=True,
            data_dir=FLAGS.data_dir,
            sentence_piece_vocab_file=sentence_piece_vocab_file,
            max_length=max_sequence_length,
        )

    # Evaluation.
    def evaler_config(split_name: str, max_examples: Optional[int] = None):
        if max_examples is None:
            max_examples = FLAGS.max_eval_examples
        elif FLAGS.max_eval_examples is not None:
            max_examples = min(max_examples, FLAGS.max_eval_examples)
        if FLAGS.data_dir == "FAKE":
            evaler_input = FakeLmInput.default_config().set(
                global_batch_size=eval_batch_size,
                total_num_batches=80,
                source_length=max_sequence_length,
                max_token_id=max_token_id,
            )
        else:
            evaler_input = LmInput.default_config().set(
                dataset_name="c4/en:3.0.1",
                split=split_name + f"[:{max_examples}]" if max_examples else split_name,
                global_batch_size=eval_batch_size,
                is_training=False,
                data_dir=FLAGS.data_dir,
                sentence_piece_vocab_file=sentence_piece_vocab_file,
                max_length=max_sequence_length,
            )
        evaler_cfg = SpmdEvaler.default_config().set(
            input=evaler_input, run_every_n_steps=steps_per_snapshot
        )
        return evaler_cfg

    cfg.evalers = dict(
        eval_train=evaler_config("train", 8192), eval_validation=evaler_config("validation")
    )

    # Summaries and checkpoints.
    cfg.checkpointer.write_every_n_steps = steps_per_snapshot
    steps_per_epoch = num_train_examples // train_batch_size
    cfg.checkpointer.keep_every_n_steps = steps_per_epoch - (steps_per_epoch % steps_per_snapshot)
    cfg.summary_writer.write_every_n_steps = 100
    return cfg

"""A skeleton trainer for ImageNet."""
import logging
from typing import Optional

import config as config_lib
from input_image import FakeImagenetInput, ImagenetInput
from trainer import SpmdEvaler, SpmdTrainer


def make_split(name: str, max_examples_per_split: Optional[int]) -> str:
    if max_examples_per_split is None:
        return name
    return f"{name}[:{max_examples_per_split}]"


def base_trainer_config(
    data_dir: str, train_batch_size: int, eval_batch_size: int
) -> config_lib.InstantiableConfig:
    """Returns a base ImageNet trainer config.

    Args:
        data_dir: the TFDS data directory. If 'FAKE', uses fake data.
        train_batch_size: the global batch size for training.
        eval_batch_size: the global batch size for evaluation. Must evenly divide all evaluation sets.

    Returns:
        A SpmdTrainer config, containing:
        - Training input config to load examples from the train split.
        - A summary writer to save training summaries every 100 steps.
        - Two evaluator configs to run every epoch and save summaries:
          * eval_train: the first 50K samples from the train split (or 2 fake batches if data_dir = "FAKE").
          * eval_validation: the validation split
        - A checkpointer config to save a ckpt every epoch.
    """
    num_train_examples = 1_281_167
    train_split = "train"

    steps_per_epoch = num_train_examples // train_batch_size

    cfg = SpmdTrainer.default_config()
    cfg.name = "imagenet_trainer"

    # Training inputs.
    read_parallelism = 2
    if data_dir == "FAKE":
        logging.warning("Using FAKE inputs!")
        cfg.input = FakeImagenetInput.default_config().set(
            global_batch_size=train_batch_size,
        )
    else:
        cfg.input = ImagenetInput.default_config().set(
            split=train_split,
            is_training=True,
            global_batch_size=train_batch_size,
            data_dir=data_dir,
            read_parallelism=read_parallelism,
            decode_parallelism=4,
            process_parallelism=64,
            prefetch_buffer_size=4,
            shuffle_buffer_size=read_parallelism * 1024,
        )

    # Evaluation.
    def evaler_config(split_name: str, max_examples: Optional[int] = None):
        if data_dir == "FAKE":
            evaler_input = FakeImagenetInput.default_config().set(
                global_batch_size=eval_batch_size,
                total_num_batches=2,
            )
        else:
            evaler_input = ImagenetInput.default_config().set(
                split=make_split(split_name, max_examples),
                is_training=False,
                global_batch_size=eval_batch_size,
                prefetch_buffer_size=4,
                data_dir=data_dir,
            )
        evaler_cfg = SpmdEvaler.default_config().set(
            input=evaler_input, run_every_n_steps=steps_per_epoch
        )
        return evaler_cfg

    cfg.evalers = dict(
        eval_train=evaler_config("train", 50000), eval_validation=evaler_config("validation")
    )

    # Summaries and checkpoints.
    cfg.checkpointer.write_every_n_steps = steps_per_epoch
    cfg.checkpointer.keep_every_n_steps = cfg.checkpointer.write_every_n_steps * 10
    cfg.summary_writer.write_every_n_steps = 100
    return cfg

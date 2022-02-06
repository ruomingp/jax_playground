"""A launcher to train ResNet-18 on ImageNet.

On the TPU VM:
    gs_bucket=permanent-us-central1-q5loch
    dir=gs://${gs_bucket}/${USER}/experiments/imagenet-$(date +%F).c
    data_dir=gs://${gs_bucket}/tensorflow_datasets
    python3 imagenet.py --trainer_dir=$dir --data_dir=$data_dir 2>&1 | tee log-$(date +%F-%T)

Or for debugging:
    python3 imagenet.py \
        --trainer_dir=${dir}_debug --data_dir=$data_dir --max_train_examples=1024 --max_eval_examples=160 \
        2>&1 | tee log

with FAKE inputs:
    python3 imagenet.py --trainer_dir=${dir}_fake --data_dir=FAKE 2>&1 | tee log

On your local machine:
    pip install tensorflow tbp-nightly
    gcloud auth application-default login
    tensorboard --logdir=$dir/summaries
"""
import logging
from typing import Optional

from absl import app, flags

import config as config_lib
import launch
import learner
import resnet
import schedule
from input_image import ImagenetInput, FakeImagenetInput
from trainer import SpmdEvaler, SpmdTrainer

flags.DEFINE_string(
    "data_dir",
    None,
    "The tfds directory. If None, uses ~/tensorflow_datasets. If 'FAKE', uses fake inputs.",
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


def make_split(name: str, max_examples_per_split: Optional[int]) -> str:
    if max_examples_per_split is None:
        return name
    return f"{name}[:{max_examples_per_split}]"


def imagenet_trainer_config() -> config_lib.InstantiableConfig:
    num_train_examples = 1_281_167
    train_split = "train"
    if FLAGS.max_train_examples is not None:
        num_train_examples = min(FLAGS.max_train_examples, num_train_examples)
        train_split = make_split(train_split, num_train_examples)

    train_batch_size = 256
    eval_batch_size = 80  # divides 50_000 and can be divided by number of devices (8)
    steps_per_epoch = num_train_examples // train_batch_size

    cfg = SpmdTrainer.default_config()
    cfg.name = "imagenet_trainer"

    # Model and optimization.
    cfg.model = resnet.ResNetModel.resnet18_config().set(num_blocks_per_stage=[4, 4, 4, 4, 4])
    learning_rate = config_lib.config_for_function(schedule.stepwise).set(
        sub=[0.1, 0.01, 0.001],
        start_step=[steps_per_epoch * 30, steps_per_epoch * 60],
    )
    cfg.learner = learner.Learner.default_config().set(
        optimizer=config_lib.config_for_function(learner.sgd_optimizer).set(
            learning_rate=learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        ),
    )
    cfg.max_step = steps_per_epoch * 90

    # Training inputs.
    read_parallelism = 1
    if FLAGS.data_dir == "FAKE":
        logging.warning("Using FAKE inputs!")
        cfg.input = FakeImagenetInput.default_config().set(
            global_batch_size=train_batch_size,
        )
    else:
        cfg.input = ImagenetInput.default_config().set(
            split=train_split,
            is_training=True,
            global_batch_size=train_batch_size,
            data_dir=FLAGS.data_dir,
            read_parallelism=read_parallelism,
            decode_parallelism=4,
            process_parallelism=64,
            prefetch_buffer_size=4,
            shuffle_buffer_size=read_parallelism * 1024,
        )

    # Evaluation.
    def evaler_config(split_name: str, max_examples: Optional[int] = None):
        if max_examples is None:
            max_examples = FLAGS.max_eval_examples
        elif FLAGS.max_eval_examples is not None:
            max_examples = min(max_examples, FLAGS.max_eval_examples)
        if FLAGS.data_dir == "FAKE":
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
                data_dir=FLAGS.data_dir,
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


def main(argv):
    launch.launch_trainer(imagenet_trainer_config())


if __name__ == "__main__":
    app.run(main)

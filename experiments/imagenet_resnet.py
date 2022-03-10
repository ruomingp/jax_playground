"""ResNet on ImageNet trainer configs.

$ cfg=ResNet-18; tpu_type=v4-8; \
  gs_bucket=permanent-us-central1-q5loch; data_dir=gs://${gs_bucket}/tensorflow_datasets; \
  exp=${cfg/\//}-$(date +%F); dir=gs://${gs_bucket}/${USER}/experiments/imagenet_resnet/${exp}; \
  taskname=${USER}-$(echo ${exp} | tr '[:upper:]' '[:lower:]')-$(date '+%H-%M'); \
  python3 -m gcp.launch --taskname=${taskname} --tpu_type=${tpu_type} --cleanup_vm --cleanup_tpu -- \
  python3 launch_trainer_main.py --config=${cfg} --trainer_dir=$dir --data_dir=$data_dir --jax_backend=tpu

View with tensorboard:
$ tensorboard --logdir=gs://permanent-us-central1-q5loch/${USER}/experiments/imagenet_resnet
"""

import config as config_lib
import learner
import optimizers
import resnet
import schedule
from experiments.imagenet_trainer import base_trainer_config


def _trainer_config(data_dir: str, train_batch_size=256) -> config_lib.InstantiableConfig:
    cfg = base_trainer_config(
        data_dir=data_dir, train_batch_size=train_batch_size, eval_batch_size=80
    )
    steps_per_epoch = cfg.checkpointer.write_every_n_steps

    # Model and optimization.
    learning_rate = config_lib.config_for_function(schedule.stepwise).set(
        sub=[0.1, 0.01, 0.001],
        start_step=[steps_per_epoch * 30, steps_per_epoch * 60],
    )
    cfg.learner = learner.Learner.default_config().set(
        optimizer=config_lib.config_for_function(optimizers.sgd_optimizer).set(
            learning_rate=learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        ),
    )
    cfg.max_step = steps_per_epoch * 90
    return cfg


def _get_config_fn(model_config, **kwargs):
    def config_fn(data_dir):
        cfg = _trainer_config(data_dir=data_dir, **kwargs)
        cfg.model = model_config
        return cfg

    return config_fn


def named_trainer_configs():
    return {
        "ResNet-Test": _get_config_fn(
            resnet.ResNetModel.resnet18_config().set(hidden_dim=4, num_blocks_per_stage=[2, 1]),
            train_batch_size=2,
        ),
        "ResNet-18": _get_config_fn(resnet.ResNetModel.resnet18_config()),
    }

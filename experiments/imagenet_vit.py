"""ViT on ImageNet trainer configs.

Reference: https://arxiv.org/pdf/2010.11929.pdf.
TODO(rpang): try DeiT: https://arxiv.org/pdf/2012.12877.pdf.

B16-adamw: gs://permanent-us-central1-q5loch/rpang/experiments/imagenet_vit_b16-2022-02-23b
L16-adamw: gs://permanent-us-central1-q5loch/rpang/experiments/imagenet_vit_L16-2022-02-24

Launch training:
$ cfg=ViT-B16-adamw; tpu_type=v4-128; \
  gs_bucket=permanent-us-central1-q5loch; data_dir=gs://${gs_bucket}/tensorflow_datasets; \
  exp=${cfg/\//}-$(date +%F); dir=gs://${gs_bucket}/${USER}/experiments/imagenet_vit/${exp}; \
  taskname=${USER}-$(echo ${exp} | tr '[:upper:]' '[:lower:]')-$(date '+%H-%M'); \
  python3 -m gcp.launch --taskname=${taskname} --tpu_type=${tpu_type} --cleanup_vm --cleanup_tpu -- \
  python3 launch_trainer_main.py --config=${cfg} --trainer_dir=$dir --data_dir=$data_dir --jax_backend=tpu

$ cfg=ViT-B16-adafactor; tpu_type=v4-128; ...
$ cfg=ViT-L16-adafactor; tpu_type=v4-128; ...

To train a G/14 model:
$ cfg=ViT-G14-adafactor; tpu_type=v4-512; ...

Test16 with FAKE inputs:
$ cfg=ViT-Test16-adamw; tpu_type=v4-8; \
  gs_bucket=permanent-us-central1-q5loch; data_dir=FAKE; \
  ...

View with tensorboard:
$ tensorboard --logdir=gs://permanent-us-central1-q5loch/${USER}/experiments/imagenet_vit
"""

from absl import logging
from optax import cosine_decay_schedule

import learner
import optimizers
import schedule
from config import InstantiableConfig, config_for_function
from experiments.imagenet_trainer import base_trainer_config
from vision_transformer import named_model_configs


def cosine_learning_rate(peak_lr, *, max_step, warmup_steps=500):
    return config_for_function(schedule.stepwise).set(
        sub=[
            config_for_function(schedule.polynomial).set(
                begin_step=0, begin_value=0, end_step=warmup_steps, end_value=peak_lr
            ),
            config_for_function(cosine_decay_schedule).set(
                init_value=peak_lr,
                decay_steps=max_step - warmup_steps,
            ),
        ],
        start_step=[warmup_steps],
    )


def _trainer_config(
    *,
    data_dir: str,
    train_batch_size: int = 4096,
    optimizer_type: str = "adamw",
    learning_rate: float = 3e-3,
) -> InstantiableConfig:
    cfg = base_trainer_config(
        data_dir=data_dir, train_batch_size=train_batch_size, eval_batch_size=256
    )

    steps_per_epoch = cfg.checkpointer.write_every_n_steps
    # Table 3.
    cfg.max_step = 300 * steps_per_epoch

    if optimizer_type == "adamw":
        learning_rate = cosine_learning_rate(
            peak_lr=learning_rate, max_step=cfg.max_step, warmup_steps=500
        )
        optimizer = config_for_function(optimizers.chain).set(
            args=[
                config_for_function(optimizers.clip_by_global_norm).set(max_norm=1),
                # TODO: support Adafactor as in https://arxiv.org/pdf/2106.04560.pdf.
                config_for_function(optimizers.adamw_optimizer).set(
                    learning_rate=learning_rate,
                    # Section 4.1: "We train all models, including ResNets, using Adam (Kingma & Ba,
                    # 2015) with β1 = 0.9, β2 = 0.999, a batch size of 4096 and apply a high weight decay of 0.1, which
                    # we found to be useful for transfer of all models"
                    b1=0.9,
                    b2=0.999,
                    # Table 3.
                    weight_decay=0.3,
                ),
            ]
        )
    elif optimizer_type == "adafactor":
        # Ref: https://arxiv.org/pdf/2106.04560.pdf.
        optimizer = config_for_function(optimizers.adafactor_optimizer).set(
            # Appendix B: "We use reciprocal square-root schedule with a linear learning rate warmup of 10k steps."
            learning_rate=config_for_function(schedule.adafactor).set(
                scale=1,
                warmup_steps=10_000,
            ),
            factored=True,
            multiply_by_parameter_scale=False,  # The paper advises setting this to false.
            clipping_threshold=1.0,
            momentum=0.9,
            dtype_momentum=cfg.model.dtype,
            # adafactor_optimizer does not scale weight decay by the learning rate, so we need to scale it
            # ourselves.
            weight_decay_rate=0.3 * (cfg.max_step**-0.5),
        )
    else:
        raise NotImplementedError(optimizer_type)

    cfg.learner = learner.Learner.default_config().set(optimizer=optimizer)
    return cfg


def _get_config_fn(config_name, model_config, **kwargs):
    def config_fn(data_dir):
        logging.info("%s: %s", config_name, kwargs)
        cfg = _trainer_config(**kwargs, data_dir=data_dir)
        cfg.model = model_config
        return cfg

    return config_fn


def named_trainer_configs():
    config_map = {}
    for model_name, model_config in named_model_configs().items():

        for optimizer_type in ("adamw", "adafactor"):
            kwargs = {
                "Test16": dict(train_batch_size=16, learning_rate=1e-4),
                "G14": dict(train_batch_size=32 * 1024),
            }.get(model_name, {})

            config_name = f"ViT-{model_name}-{optimizer_type}"
            config_map[config_name] = _get_config_fn(config_name, model_config, **kwargs)

    return config_map

"""A launcher to train GPT2 on c4-en.

Example summaries:
- gs://permanent-us-central2-0rxn/tom_gunter/experiments/c4_gpt2-2022-03-02_4
- gs://permanent-us-central2-0rxn/rpang/experiments/c4_gpt2/c4-gpt2-adafactor-2022-03-07-warmup2k/

Partial reference (Training configuration omitted from GPT2 paper, partially described in GPT, GPT3 provides more detail):
RefA: GPT - <https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf>
RefB: GPT2 - <http://www.persagen.com/files/misc/radford2019language.pdf>.
RefC: GPT3 - <https://arxiv.org/abs/2005.14165>

$ gs_bucket=permanent-us-central2-0rxn; \
  dir=gs://${gs_bucket}/${USER}/experiments/c4_gpt2-$(date +%F); \
  data_dir=gs://${gs_bucket}/tensorflow_datasets;

To test on v4-8:
$ python3 -m gcp.launch --tpu_type=v4-8 --cleanup_vm --cleanup_tpu -- \
  python3 c4_gpt2.py --trainer_dir=$dir --data_dir=$data_dir --jax_backend=tpu --global_batch_size=64

with FAKE inputs:
$ python3 -m gcp.launch --tpu_type=v4-8 --cleanup_vm --cleanup_tpu -- \
  python3 c4_gpt2.py --trainer_dir=${dir}_fake --data_dir=FAKE --max_train_examples=1048576 --jax_backend=tpu --global_batch_size=64


To train a gpt2 model:
$ exp=c4-gpt2-adamw-$(date +%F); dir=gs://${gs_bucket}/${USER}/experiments/c4_gpt2/${exp}; \
  python3 -m gcp.launch --taskname=${exp}-$(date '+%H-%M') --tpu_type=v4-64 --cleanup_vm --cleanup_tpu -- \
  python3 c4_gpt2.py --optmizer_type=adamw --trainer_dir=$dir --data_dir=$data_dir --jax_backend=tpu

$ exp=c4-gpt2-adafactor-$(date +%F)-warmup2k; dir=gs://${gs_bucket}/${USER}/experiments/c4_gpt2/${exp}; \
  python3 -m gcp.launch --taskname=${exp}-$(date '+%H-%M') --tpu_type=v4-64 --cleanup_vm --cleanup_tpu -- \
  python3 c4_gpt2.py --optimizer_type=adafactor --trainer_dir=$dir --data_dir=$data_dir --jax_backend=tpu

Adam vs. Adafactor comparison:
% tensorboard --logdir=gs://permanent-us-central2-0rxn/rpang/experiments/c4_gpt2

To train a gpt2-xl model:
$ dir=gs://${gs_bucket}/${USER}/experiments/c4_gpt2-xl-$(date +%F)
$ python3 -m gcp.launch --tpu_type=v4-128 --cleanup_vm --cleanup_tpu -- \
  python3 c4_gpt2.py model=gpt2-xl --trainer_dir=$dir --data_dir=$data_dir --jax_backend=tpu

To train a gpt2-7b model (sharded 2 ways):
$ dir=gs://${gs_bucket}/${USER}/experiments/c4_gpt2-6_7b-$(date +%F)
$ python3 -m gcp.launch --tpu_type=v4-256 --cleanup_vm --cleanup_tpu -- \
  python3 c4_gpt2.py model=gpt2-6_7b --trainer_dir=$dir --data_dir=$data_dir --jax_backend=tpu \
  --mesh_shape=32,4

To train a gpt2-10b model (sharded 8 ways):
$ dir=gs://${gs_bucket}/${USER}/experiments/c4_gpt2-10b-$(date +%F)
$ python3 -m gcp.launch --tpu_type=v4-512 --cleanup_vm --cleanup_tpu -- \
  python3 c4_gpt2.py --model=gpt2-10b --trainer_dir=$dir --data_dir=$data_dir --jax_backend=tpu \
  --mesh_shape=64,4

View with tensorboard:
$ tensorboard --logdir=$dir/summaries
"""
# Import jax before anything else to avoid problems such as:
# tpu_library_init_fns.inc:98] TpuEmbeddingEngine_ExecutePartitioner not available in this library.

import jax

print(f"jax version={jax.__version__}")

import jax.numpy as jnp
from absl import app, flags
from optax import cosine_decay_schedule

import causal_lm
import launch
import learner
import optimizers
import schedule
from attention import PipelinedTransformerLayer, RepeatedTransformerLayer, StackedTransformerLayer
from c4_lm_trainer import base_trainer_config
from config import InstantiableConfig, config_for_class, config_for_function
from module import BaseLayer
from param_init import GaussianInitializer

NAMED_GPT2_LMS = {
    # 111,008,256 params; Ref B. Table 2, Ref A.
    "gpt2-small": dict(
        num_layers=12,
        hidden_dim=768,
        num_heads=12,
        dtype=jnp.float32,
        dropout_rate=0.1,
    ),
    # 1,529,628,800 params; Ref B. Table 2; Ref C Table 2.1.
    # The exact GPT2-XL model config (Ref A, section 4.1) uses dropout and less weight-decay.
    # We favor the config from Ref C, as removing dropout saves significant memory.
    "gpt2-xl": dict(
        num_layers=48,
        hidden_dim=1600,
        num_heads=25,
        dtype=jnp.bfloat16,
        dropout_rate=0.0,
    ),
    # 6,582,575,104 params; Ref C Table 2.1.
    "gpt2-6_7b": dict(
        num_layers=32,
        hidden_dim=4096,
        num_heads=32,
        dtype=jnp.bfloat16,
        dropout_rate=0.0,  # Ref C: Appendix B (only weight decay mentioned as regularization).
    ),
    # 9,804,652,544 params; ~Ref C Table 2.1 for some params, but this is not a published model.
    "gpt2-10b": dict(
        num_layers=48,
        hidden_dim=4096,
        num_heads=32,
        dtype=jnp.bfloat16,
        dropout_rate=0.0,
    ),
}

NAMED_GPT2_OPTIMS = {
    "gpt2-small": dict(  # Ref C Table 2.1. # Ref A Section 4.1.
        lr=6e-4,
        weight_decay=0.01,
    ),
    "gpt2-xl": dict(  # Ref C Table 2.1. # Ref C: Appendix B.
        lr=2e-4,
        weight_decay=0.1,
    ),
    "gpt2-6_7b": dict(  # Ref C Table 2.1. # Ref C: Appendix B.
        lr=1.2e-4,
        weight_decay=0.1,
    ),
    "gpt2-10b": dict(  # Ref C Table 2.1. # Ref C: Appendix B.
        lr=1e-4,
        weight_decay=0.1,
    ),
}

TRANSFORMER_STACK_ALTS = {
    "stacked": StackedTransformerLayer,
    "repeated": RepeatedTransformerLayer,
    "pipelined": PipelinedTransformerLayer,
}

flags.DEFINE_enum("model", "gpt2-small", NAMED_GPT2_LMS.keys(), "The GPT2 LM model name.")
flags.DEFINE_enum(
    "transformer_type",
    "stacked",
    TRANSFORMER_STACK_ALTS.keys(),
    "The type of transformer stack to use when constructing the model; "
    "analytically identical, choice affects compilation and/or parallelism strategy.",
)
flags.DEFINE_enum("optimizer_type", "adamw", ("adamw", "adafactor"), "The optimizer type.")
flags.DEFINE_integer("global_batch_size", 512, "The global batch size for training.")
# N.B. The global batch size may need to be increased for the larger models with max seq len of 1024.

FLAGS = flags.FLAGS


def trainer_config() -> InstantiableConfig:
    max_sequence_length = 1024  # Ref B: Section 2.4; GPT3 increases to 2048, Ref C Section 2.
    cfg = base_trainer_config(
        train_batch_size=FLAGS.global_batch_size,
        eval_batch_size=FLAGS.global_batch_size,
        max_sequence_length=max_sequence_length,
    )
    cfg.model = gpt2_model_cfg(
        FLAGS.model,
        vocab_size=32_768,  # Must be > the size of the tokenizer vocab.
        source_length=max_sequence_length,
        transformer_cls=TRANSFORMER_STACK_ALTS[FLAGS.transformer_type],
    )

    approx_steps_per_epoch = cfg.checkpointer.keep_every_n_steps
    cfg.max_step = approx_steps_per_epoch
    # TODO(tom_gunter): Tune training length depending on dataset size & contents.
    # c4-en is ~50x larger than WebText used for GPT2, and one epoch should be >
    # the 300B tokens seen for each of the GPT3 models (Ref C. Table 2.1).
    warmup_steps = 2000  # Ref A Section 4.1, Ref C Appendix B.
    named_optim_config = NAMED_GPT2_OPTIMS[FLAGS.model]
    peak_lr = named_optim_config["lr"]
    learning_rate = config_for_function(schedule.stepwise).set(
        sub=[
            # Ref A: Section 4.1.
            config_for_function(schedule.polynomial).set(
                begin_step=0,
                begin_value=0,
                end_step=warmup_steps,
                end_value=peak_lr,
            ),
            # From GPT-3 (https://arxiv.org/pdf/2005.14165.pdf) appendix B:
            # We use cosine decay for learning rate down to 10% of its value, over 260 billion tokens (after 260
            # billion tokens, training continues at 10% of the original learning rate)
            config_for_function(cosine_decay_schedule).set(
                init_value=peak_lr,
                decay_steps=cfg.max_step - warmup_steps,
                alpha=0.1,
            ),
        ],
        start_step=[warmup_steps],
    )
    if FLAGS.optimizer_type == "adamw":
        optimizer = config_for_function(optimizers.chain).set(
            args=[
                # Ref C: Appendix B.
                config_for_function(optimizers.clip_by_global_norm).set(max_norm=1),
                config_for_function(optimizers.adamw_optimizer).set(
                    learning_rate=learning_rate,
                    b1=0.9,
                    b2=0.95,  # Ref C: Appendix B.
                    eps=1e-8,
                    weight_decay=named_optim_config["weight_decay"],
                ),
            ]
        )
    elif FLAGS.optimizer_type == "adafactor":
        optimizer = config_for_function(optimizers.adafactor_optimizer).set(
            learning_rate=config_for_function(schedule.adafactor).set(
                scale=1, warmup_steps=warmup_steps
            ),
            clipping_threshold=1.0,
            momentum=0.9,
            dtype_momentum=cfg.model.dtype,
            # adafactor_optimizer does not scale weight decay by the learning rate, so we need to scale it
            # ourselves.
            weight_decay_rate=1e-4 * named_optim_config["weight_decay"],
        )
    else:
        raise NotImplementedError(FLAGS.optimizer_type)
    cfg.learner = learner.Learner.default_config().set(optimizer=optimizer)
    return cfg


def gpt2_model_cfg(
    name: str,
    vocab_size: int,
    source_length: int,
    transformer_cls: BaseLayer,
):
    if name not in NAMED_GPT2_LMS:
        raise ValueError(f"name {name} not in {NAMED_GPT2_LMS.keys()}")
    gpt2_kwargs = NAMED_GPT2_LMS[name]
    gpt2_model_cfg = causal_lm.Model.default_config()
    gpt2_model_cfg.dropout_rate = gpt2_kwargs.pop("dropout_rate")
    gpt2_model_cfg.vocab_size = vocab_size
    gpt2_model_cfg.source_length = source_length
    hidden_dim = gpt2_kwargs["hidden_dim"]
    gpt2_model_cfg.hidden_dim = hidden_dim
    # Set embedding initialization.
    # https://github.com/openai/gpt-2/blob/c2dae27/src/model.py#L154
    gpt2_model_cfg.emb.param_init = config_for_class(GaussianInitializer).set(std=0.02)
    # https://github.com/openai/gpt-2/blob/c2dae27/src/model.py#L152
    gpt2_model_cfg.pos_emb.param_init = config_for_class(GaussianInitializer).set(std=0.01)
    gpt2_model_cfg.dtype = gpt2_kwargs.pop("dtype")
    gpt2_model_cfg.transformer = causal_lm.gpt2_transformer_cfg(
        **gpt2_kwargs, transformer_cls=transformer_cls
    )
    return gpt2_model_cfg


def main(_):
    launch.launch_trainer(trainer_config())


if __name__ == "__main__":
    app.run(main)

"""A library with common flags to launch a trainer."""
import copy
from typing import Any

import jax  # jax must be imported before tensorflow!
from absl import flags, logging

import config as config_lib
from trainer import SpmdTrainer

flags.DEFINE_string(
    "trainer_dir",
    None,
    "The root directory of the trainer. "
    "Checkpoints will be stored in <dir>/checkpoints. "
    "Summaries will be stored in <dir>/summaries.",
    required=True,
)
flags.DEFINE_integer(
    "trainer_prng_seed",
    0,
    "The seed for jax.random.PRNGKey(). "
    "Used for initializing model parameters and pseudo-random number generation during training.",
)
flags.DEFINE_integer("jax_profiler_port", None, "If not None, the profiler port.")

FLAGS = flags.FLAGS


def launch_trainer(trainer_config: config_lib.InstantiableConfig) -> Any:
    trainer_config = copy.deepcopy(trainer_config)
    trainer_config.dir = trainer_config.dir or FLAGS.trainer_dir
    logging.info("Trainer config:\n%s", trainer_config.debug_string())

    if FLAGS.jax_profiler_port is not None:
        # Start jax.profiler for Tensorboard and profiling in open source.
        jax.profiler.start_server(FLAGS.jax_profiler_port)

    trainer: SpmdTrainer = trainer_config.instantiate(parent=None)
    prng_key = jax.random.PRNGKey(seed=FLAGS.trainer_prng_seed)
    return trainer.run(prng_key)

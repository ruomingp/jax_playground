"""A library with common flags to launch a trainer."""
import copy
import logging as pylogging
import os
from typing import Any

import jax  # jax must be imported before tensorflow!
from absl import flags, logging
from tensorflow.io.gfile import GFile

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
flags.DEFINE_string(
    "jax_backend", None, "If not None, ensures that trainer runs on the specified XLA backend."
)
flags.DEFINE_list(
    "mesh_axis_names",
    None,
    "The default device mesh axis names (if the trainer config does not specify this explicitly).",
)
flags.DEFINE_list(
    "mesh_shape",
    None,
    "If not None, a list of integers describing the default shape of the device mesh. "
    "Used by SpmdTrainer config if it is not set in the config explicitly. "
    "The number of dimensions must match the number of elements in --mesh_axis_names. "
    "If None, defaults to [jax.device_count(), 1], "
    "which represents a data-parallel mesh with the default mesh axis names ('data', 'model').",
)

FLAGS = flags.FLAGS


def setup_trainer_config(
    trainer_config: config_lib.InstantiableConfig,
) -> config_lib.InstantiableConfig:
    trainer_config = copy.deepcopy(trainer_config)
    trainer_config.dir = trainer_config.dir or FLAGS.trainer_dir
    trainer_config.mesh_axis_names = (
        trainer_config.mesh_axis_names or FLAGS.mesh_axis_names or ("data", "model")
    )
    mesh_shape_from_flags = (
        None if FLAGS.mesh_shape is None else [int(dim) for dim in FLAGS.mesh_shape]
    )
    trainer_config.mesh_shape = (
        trainer_config.mesh_shape or mesh_shape_from_flags or (len(jax.devices()), 1)
    )
    return trainer_config


def launch_trainer(trainer_config: config_lib.InstantiableConfig) -> Any:
    logging.get_absl_handler().addFilter(InfoLogOnlyOnMaster())
    # Use a GSPMD-friendly PRNG implementation.
    jax.config.update("jax_default_prng_impl", "rbg")

    if FLAGS.jax_profiler_port is not None:
        # Start jax.profiler for Tensorboard and profiling in open source.
        jax.profiler.start_server(FLAGS.jax_profiler_port)

    devices = jax.devices()
    logging.info("Devices: %s", devices)
    if FLAGS.jax_backend is not None:
        if not devices or not all(device.platform == FLAGS.jax_backend for device in devices):
            raise RuntimeError(f"Expected backend {FLAGS.jax_backend}. Got {devices}.")

    trainer_config = setup_trainer_config(trainer_config)
    trainer_config_debug_string = trainer_config.debug_string()
    logging.info("Trainer config:\n%s", trainer_config_debug_string)
    if jax.process_index() == 0:
        with GFile(os.path.join(trainer_config.dir, "trainer_config"), "w") as f:
            f.write(trainer_config_debug_string)

    trainer: SpmdTrainer = trainer_config.instantiate(parent=None)
    prng_key = jax.random.PRNGKey(seed=FLAGS.trainer_prng_seed)
    return trainer.run(prng_key)


class InfoLogOnlyOnMaster(pylogging.Filter):
    # Filter to only log levels >= logging.INFO if on master process.

    def __init__(self, name=""):
        super().__init__(name=name)
        self._jax_pid = jax.process_index()

    def filter(self, record):
        if self._jax_pid != 0:
            return record.levelno < logging.INFO
        return True

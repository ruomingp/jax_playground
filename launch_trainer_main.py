import sys

# Import jax before anything else to avoid problems such as:
# tpu_library_init_fns.inc:98] TpuEmbeddingEngine_ExecutePartitioner not available in this library.
import jax

print(f"jax version={jax.__version__}, devices={jax.devices()}", file=sys.stderr)

from absl import app, flags

import launch
from experiments import imagenet_resnet, imagenet_vit


def named_trainer_configs():
    config_map = {}
    config_map.update(imagenet_resnet.named_trainer_configs())
    config_map.update(imagenet_vit.named_trainer_configs())
    return config_map


flags.DEFINE_enum(
    "config", None, list(named_trainer_configs().keys()), "The trainer config name.", required=True
)
flags.DEFINE_string(
    "data_dir",
    None,
    "The tfds directory. If None, uses ~/tensorflow_datasets. If 'FAKE', uses fake inputs.",
)

FLAGS = flags.FLAGS


def main(argv):
    trainer_config_fn = named_trainer_configs().get(FLAGS.config)
    launch.launch_trainer(trainer_config_fn(data_dir=FLAGS.data_dir))


if __name__ == "__main__":
    app.run(main)

from image import ImagenetInput
import tensorflow_datasets as tfds

cfg = ImagenetInput.default_config().set(
    name="benchmark", split="train", is_training=True,
    global_batch_size=256, data_dir="gs://permanent-us-central1-q5loch/tensorflow_datasets")
inputs = cfg.instantiate(parent=None)
print(tfds.benchmark(inputs._ds, batch_size=256))

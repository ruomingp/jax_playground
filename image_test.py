from absl import logging
from absl.testing import absltest
from image import ImagenetInput
import utils


def _count_batches(dataset, max_batches=100):
    num_batches = 0
    for batch in dataset:
        num_batches += 1
        if num_batches > max_batches:
            return -1
    return num_batches


class ImagenetInputTest(absltest.TestCase):

    def testIteration(self):
        cfg = ImagenetInput.default_config().set(
            name="test", tfds_name="imagenet2012", split="train[:40]", global_batch_size=8, is_training=False,
            data_dir="gs://permanent-us-central1-q5loch/tensorflow_datasets",
        )
        dataset = cfg.instantiate(parent=None)
        # 40 images / 8.
        self.assertEqual(5, _count_batches(dataset))
        for batch in dataset:
            self.assertEqual({'image': (8, 224, 224, 3), 'label': (8,)}, utils.shapes(batch))
            break

    def testIndivisible(self):
        cfg = ImagenetInput.default_config().set(
            name="test", tfds_name="imagenet2012", split="train[:31]", global_batch_size=8, is_training=False,
            data_dir="gs://permanent-us-central1-q5loch/tensorflow_datasets",
        )
        with self.assertRaisesRegex(ValueError, "must be divisible by global_batch_size"):
            cfg.instantiate(parent=None)


if __name__ == "__main__":
    absltest.main()

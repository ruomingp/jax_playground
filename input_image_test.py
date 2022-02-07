from absl import logging
from absl.testing import absltest, parameterized

import utils
from input_image import ImagenetInput


def _count_batches(dataset, max_batches=100):
    num_batches = 0
    for _ in dataset:
        num_batches += 1
        if num_batches > max_batches:
            return -1
    return num_batches


class ImagenetInputTest(parameterized.TestCase):
    @parameterized.parameters(False, True)
    def testIteration(self, is_training):
        cfg = ImagenetInput.default_config().set(
            name="test",
            dataset_name="imagenet2012",
            split="train[:40]",
            global_batch_size=8,
            is_training=is_training,
            data_dir="gs://permanent-us-central1-q5loch/tensorflow_datasets",
        )
        dataset = cfg.instantiate(parent=None)
        if is_training:
            # For training, we loop over the dataset forever.
            self.assertEqual(-1, _count_batches(dataset, max_batches=24))
        else:
            # For training, we loop over the dataset only once.
            self.assertEqual(40 // 8, _count_batches(dataset, max_batches=100))
        for batch in dataset:
            self.assertEqual({"image": (8, 224, 224, 3), "label": (8,)}, utils.shapes(batch))
            break

    def testIndivisible(self):
        cfg = ImagenetInput.default_config().set(
            name="test",
            dataset_name="imagenet2012",
            split="train[:31]",
            global_batch_size=8,
            is_training=False,
            data_dir="gs://permanent-us-central1-q5loch/tensorflow_datasets",
        )
        with self.assertRaisesRegex(ValueError, "must be divisible by global_batch_size"):
            cfg.instantiate(parent=None)


if __name__ == "__main__":
    absltest.main()

from absl import logging
from absl.testing import absltest, parameterized
from tensorflow_datasets.testing.mocking import mock_data

import utils
from input_image import ImagenetInput


def _count_batches(dataset, max_batches=100):
    for n, _ in enumerate(dataset):
        if n >= max_batches:
            return -1
    return n + 1


class ImagenetInputTest(parameterized.TestCase):
    @parameterized.parameters(False, True)
    def testIteration(self, is_training):
        with mock_data(num_examples=40):
            cfg = ImagenetInput.default_config().set(
                name="test",
                dataset_name="imagenet2012",
                split="train",
                global_batch_size=8,
                is_training=is_training,
                data_dir="gs://permanent-us-central1-q5loch/tensorflow_datasets",
            )
            dataset = cfg.instantiate(parent=None)
            if is_training:
                # For training, we loop over the dataset forever.
                self.assertEqual(-1, _count_batches(dataset, max_batches=24))
            else:
                # For evaluation, we loop over the dataset only once.
                self.assertEqual(40 // 8, _count_batches(dataset, max_batches=100))
            for batch in dataset:
                self.assertEqual({"image": (8, 224, 224, 3), "label": (8,)}, utils.shapes(batch))
                break

    def testIndivisible(self):
        with mock_data(num_examples=31):
            cfg = ImagenetInput.default_config().set(
                name="test",
                dataset_name="imagenet2012",
                split="train",
                global_batch_size=8,
                is_training=False,
                data_dir="gs://permanent-us-central1-q5loch/tensorflow_datasets",
            )
            dataset = cfg.instantiate(parent=None)
            num_batches = 0
            for batch in dataset:
                logging.info(batch["label"])
                self.assertEqual({"image": (8, 224, 224, 3), "label": (8,)}, utils.shapes(batch))
                num_batches += 1
            self.assertEqual(num_batches, 31 // 8)


if __name__ == "__main__":
    absltest.main()

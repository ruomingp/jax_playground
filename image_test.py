from absl import logging
from absl.testing import absltest
from image import ImagenetInput
import utils


class ImagenetInputTest(absltest.TestCase):
    def testIteration(self):
        cfg = ImagenetInput.default_config().set(name="imagenet", tfds_name="mnist", split="validation", batch_size=8, is_training=False)
        dataset = cfg.instantiate(parent=None)
        for batch in dataset:
            logging.info("batch=%s", utils.shapes(batch))


if __name__ == "__main__":
    absltest.main()

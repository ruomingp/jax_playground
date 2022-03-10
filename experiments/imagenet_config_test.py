from absl.testing import absltest, parameterized

from experiments import imagenet_resnet, imagenet_vit
from test_utils import TrainerConfigTestCase


class ImageNetResNetTest(TrainerConfigTestCase):
    def testTrainerConfig(self):
        self._test_with_trainer_config(
            imagenet_resnet.named_trainer_configs()[f"ResNet-Test"](data_dir="FAKE"),
        )


class ImageNetViTTest(TrainerConfigTestCase):
    @parameterized.parameters("adamw", "adafactor")
    def testTrainerConfig(self, optimizer_type):
        self._test_with_trainer_config(
            imagenet_vit.named_trainer_configs()[f"ViT-Test16-{optimizer_type}"](data_dir="FAKE"),
            mesh_size=dict(model=4),
        )


if __name__ == "__main__":
    absltest.main()

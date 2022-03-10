import difflib

import jax.random
import numpy as np
import torch
from absl import logging
from absl.testing import absltest
from jax import numpy as jnp
from torchvision.models.resnet import resnet18

import resnet
import utils
from module import NestedTensor
from module import functional as F
from resnet import ResNetModel


def params_from_conv(ref: torch.nn.Module) -> NestedTensor:
    return {
        "weight": ref.weight.permute(2, 3, 1, 0),
    }


def params_from_linear(ref: torch.nn.Module) -> NestedTensor:
    return {
        "weight": ref.weight.transpose(1, 0),
        "bias": ref.bias,
    }


def params_from_bn(ref: torch.nn.Module) -> NestedTensor:
    return {
        "scale": ref.weight,
        "bias": ref.bias,
        "moving_mean": ref.running_mean,
        "moving_variance": ref.running_var,
    }


def params_from_downsample(ref: torch.nn.ModuleList) -> NestedTensor:
    return {
        "conv": params_from_conv(ref[0]),
        "norm": params_from_bn(ref[1]),
    }


def params_from_block(ref: torch.nn.Module) -> NestedTensor:
    params = {
        "conv1": params_from_conv(ref.conv1),
        "norm1": params_from_bn(ref.bn1),
        "conv2": params_from_conv(ref.conv2),
        "norm2": params_from_bn(ref.bn2),
    }
    if getattr(ref, "downsample"):
        params["downsample"] = params_from_downsample(ref.downsample)
    return params


def params_from_stage(ref: torch.nn.ModuleList) -> NestedTensor:
    return {f"block{i}": params_from_block(block) for i, block in enumerate(ref)}


def params_from_resnet(ref: torch.nn.Module) -> NestedTensor:
    return {
        "conv1": params_from_conv(ref.conv1),
        "norm1": params_from_bn(ref.bn1),
        **{f"stage{i}": params_from_stage(getattr(ref, f"layer{i + 1}")) for i in range(4)},
        "fc": params_from_linear(ref.fc),
    }


class ResNetTest(absltest.TestCase):
    def testResNet18(self):
        model: resnet.ResNetModel = (
            ResNetModel.resnet18_config().set(name="test").instantiate(parent=None)
        )
        init_params = model.initialize_parameters_recursively(jax.random.PRNGKey(1))
        param_spec_strs = [
            f"{name}={list(param.shape)}" for name, param in utils.flatten_items(init_params)
        ]

        ref = resnet18(pretrained=True)
        for name, param in ref.named_parameters():
            logging.info("ref param: %s=%s", name, list(param.shape))

        model_params = jax.tree_map(lambda x: utils.as_tensor(x), params_from_resnet(ref))
        model_param_strs = [
            f"{name}={list(param.shape)}" for name, param in utils.flatten_items(model_params)
        ]

        self.assertEqual(
            model_param_strs,
            param_spec_strs,
            "\n".join(difflib.ndiff(model_param_strs, param_spec_strs)),
        )

        batch_size = 2
        inputs = {
            "image": np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32),
            "label": np.random.randint(0, 999, [batch_size]).astype(np.int32),
        }
        (loss, aux), _ = F(
            model,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=model_params,
            inputs=jax.tree_map(lambda x: jnp.asarray(x), inputs),
        )
        ref.eval()
        ref_logits = ref(torch.as_tensor(inputs["image"]).permute(0, 3, 1, 2)).detach().numpy()
        np.testing.assert_allclose(aux["logits"], ref_logits, atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    absltest.main()

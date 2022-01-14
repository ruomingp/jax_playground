"""Reference: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py."""
import math
from typing import Optional

import jax.nn
from jax import numpy as jnp

import config as config_lib
import layers
import param_init
from metrics import WeightedScalar
from layers import Conv2D, BatchNorm, get_activation_fn
from module import BaseLayer, Module

Tensor = jnp.ndarray


def batch_norm():
    return BatchNorm.default_config().set(decay=0.9)


class Downsample(BaseLayer):
    """A downsampling layer."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("input_dim", 0, "Input feature dim.")
        cfg.define("output_dim", 0, "Output feature dim.")
        cfg.define("stride", 1, "The convolution stride.")
        cfg.define(
            "conv",
            Conv2D.default_config().set(window=(3, 3), bias=False, param_partition_spec=(None, None, None, "model")),
            "The convolution layer config.",
        )
        cfg.define(
            "norm",
            batch_norm(),
            "The normalization layer config.",
        )
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "conv",
            cfg.conv.set(
                window=(1, 1),
                strides=(cfg.stride, cfg.stride),
                input_dim=cfg.input_dim,
                output_dim=cfg.output_dim,
            ),
        )
        self._add_child("norm", cfg.norm.set(dim=cfg.output_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.conv(inputs)
        return self.norm(x)


class BasicBlock(BaseLayer):
    """A basic ResNet block."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("input_dim", 0, "Input feature dim.")
        cfg.define("output_dim", 0, "Output feature dim.")
        cfg.define("stride", 1, "The convolution stride.")
        cfg.define(
            "conv",
            Conv2D.default_config().set(
                window=(3, 3), bias=False, padding=((1, 1), (1, 1)),
                param_partition_spec=(None, None, None, "model"),
            ),
            "The convolution layer config.",
        )
        cfg.define(
            "norm",
            batch_norm(),
            "The normalization layer config.",
        )
        cfg.define("activation", "nn.relu", "The activation function.")
        cfg.define(
            "downsample",
            None,
            "If strides > 1, the layer used to downsample the skip connection features.",
        )
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "conv1",
            cfg.conv.set(
                strides=(cfg.stride, cfg.stride),
                input_dim=cfg.input_dim,
                output_dim=cfg.output_dim,
            ),
        )
        self._add_child("norm1", cfg.norm.set(dim=cfg.output_dim))
        self._add_child(
            "conv2",
            cfg.conv.set(
                strides=(1, 1), input_dim=cfg.output_dim, output_dim=cfg.output_dim
            ),
        )
        self._add_child("norm2", cfg.norm.set(dim=cfg.output_dim))
        if cfg.downsample is not None:
            self._add_child(
                "downsample",
                cfg.downsample.set(
                    stride=cfg.stride,
                    input_dim=cfg.input_dim,
                    output_dim=cfg.output_dim,
                ),
            )

    def forward(self, inputs: Tensor) -> Tensor:
        cfg = self.config
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = get_activation_fn(cfg.activation)(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if cfg.downsample is not None:
            inputs = self.downsample(inputs)
        x += inputs
        x = get_activation_fn(cfg.activation)(x)
        return x


class ResNetStage(BaseLayer):
    """A stage of ResNet, consisting of multiple blocks."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("input_dim", 0, "Input feature dim.")
        cfg.define("output_dim", 0, "Output feature dim.")
        cfg.define("stride", 1, "The convolution stride.")
        cfg.define("block", BasicBlock.default_config(), "The block config.")
        cfg.define("num_blocks", None, "Number of blocks in this stage.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        input_dim = cfg.input_dim
        for block_i in range(cfg.num_blocks):
            if block_i == 0:
                stride = cfg.stride
            else:
                stride = 1
            if stride != 1 or input_dim != cfg.output_dim:
                downsample = Downsample.default_config().set(
                    input_dim=input_dim, output_dim=cfg.output_dim, stride=stride
                )
            else:
                downsample = None
            self._add_child(
                f"block{block_i}",
                cfg.block.set(
                    input_dim=input_dim,
                    output_dim=cfg.output_dim,
                    stride=stride,
                    downsample=downsample,
                ),
            )
            input_dim = cfg.output_dim

    def forward(self, inputs: Tensor) -> Tensor:
        cfg = self.config
        x = inputs
        for block_i in range(cfg.num_blocks):
            x = getattr(self, f"block{block_i}")(x)
        return x


class ResNetModel(BaseLayer):
    """The generic ResNet model."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define(
            "hidden_dim",
            64,
            "The feature dim between the stem layer and the first block.",
        )
        cfg.define(
            "norm",
            batch_norm(),
            "The normalization layer config.",
        )
        cfg.define("stage", ResNetStage.default_config(), "The stage config.")
        cfg.define(
            "num_blocks_per_stage",
            None,
            "A list of integers, representing number of blocks per stage.",
        )
        cfg.define(
            "num_classes",
            1000,
            "The number of classification classes.",
        )
        cfg.param_init = param_init.DefaultInitializer.default_config().set(
            # Equivalent to kaiming_normal_(mode='fan_out', nonlinearity='relu').
            fan="fan_out",
            distribution="normal",
            gain=math.sqrt(2),
        )
        cfg.dtype = jnp.float32
        return cfg

    @classmethod
    def resnet18_config(cls):
        cfg = cls.default_config()
        cfg.num_blocks_per_stage = [2, 2, 2, 2]
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        hidden_dim = cfg.hidden_dim
        self._add_child(
            "conv1",
            layers.Conv2D.default_config().set(
                window=(7, 7),
                padding=((3, 3), (3, 3)),
                strides=(2, 2),
                input_dim=3,
                output_dim=hidden_dim,
                bias=False,
                param_partition_spec=(None, None, None, "model")
            ),
        )
        self._add_child("norm1", cfg.norm.set(dim=hidden_dim))
        for stage_i, num_blocks in enumerate(cfg.num_blocks_per_stage):
            self._add_child(
                f"stage{stage_i}",
                cfg.stage.set(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim * 2,
                    stride=1 if stage_i == 0 else 2,
                    num_blocks=num_blocks,
                ),
            )
            hidden_dim *= 2
        self._add_child(
            "fc",
            layers.Linear.default_config().set(
                input_dim=hidden_dim, output_dim=cfg.num_classes, bias=True,
                param_partition_spec=("model", None)
            ),
        )

    def forward(self, image: Tensor, label: Tensor) -> Tensor:
        cfg = self.config
        image = image.astype(self.dtype())
        x = self.conv1(image)
        x = self.norm1(x)
        x = jax.nn.relu(x)
        x = jax.lax.reduce_window(
            x,
            init_value=-jnp.inf,
            computation=jax.lax.max,
            window_dimensions=(1, 3, 3, 1),
            window_strides=(1, 2, 2, 1),
            padding=((0, 0), (1, 1), (1, 1), (0, 0)),
        )
        for stage_i, num_blocks in enumerate(cfg.num_blocks_per_stage):
            x = getattr(self, f"stage{stage_i}")(x)
        # [batch, hidden].
        x = jnp.mean(x, axis=(1, 2))
        # [batch, num_classes].
        logits = self.fc(x)
        # [batch, num_classes].
        label_onehot = jax.nn.one_hot(label, cfg.num_classes, dtype=logits.dtype)
        # [batch].
        per_example_loss = jnp.sum(-label_onehot * jax.nn.log_softmax(logits), axis=-1)
        # Scalar.
        loss = jnp.mean(per_example_loss)
        # [batch].
        predictions = jnp.argmax(logits, axis=-1)
        num_examples = float(label.shape[0])
        accuracy = jnp.equal(predictions, label).sum() / num_examples
        return loss, dict(
            accuracy=WeightedScalar(accuracy, num_examples),
            loss=WeightedScalar(loss, num_examples),
        )

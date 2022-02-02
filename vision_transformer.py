"""Vision transformer layers.

References:
- https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py
"""
from typing import Tuple

import jax.nn
from jax import numpy as jnp

import config as config_lib
from attention import LearnedPositionalEmbedding, TransformerLayer
from layers import Conv2D, Dropout, LayerNorm
from metrics import WeightedScalar
from module import BaseLayer, Module, NestedTensor, Tensor
from param_init import GaussianInitializer
from resnet import ResNetStage


class TransformerSequenceEncoder(BaseLayer):
    """A sequence encoder consisting of multiple transformer layers."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("input_dim", 0, "Input feature dim.")
        cfg.define("input_dropout", Dropout.default_config(), "Input dropout config.")
        cfg.define("pos_emb", LearnedPositionalEmbedding.default_config(),
                   "Positional embedding config.")
        # https://github.com/google-research/vision_transformer/blob/dc8ddbcdeefd281d6cc7fea0c97355495688ca9c/vit_jax/models.py#L189
        cfg.pos_emb.param_init = GaussianInitializer(0.02)
        cfg.define("num_layers", 0, "The number of layers.")
        cfg.define(
            "transformer",
            TransformerLayer.default_config(),
            "The transformer layer config.",
        )
        # Vision transformer uses 'gelu' and dropout=0.1 by default.
        cfg.transformer.feed_forward.activation = 'nn.gelu'
        cfg.transformer.feed_forward.dropout.rate = 0.1
        cfg.transformer.self_attention.dropout.rate = 0.1
        cfg.transformer.self_attention.attention.dropout.rate = 0.1
        cfg.define(
            "output_norm",
            LayerNorm.default_config(),
            "The normalization layer config for encoder output.",
        )
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("input_dropout", cfg.input_dropout)
        self._add_child("pos_emb", cfg.pos_emb.set(input_dim=cfg.input_dim))
        for i in range(cfg.num_layers):
            self._add_child(f"layer{i}", cfg.transformer.set(input_dim=cfg.input_dim))
        self._add_child("output_norm", cfg.output_norm.set(input_dim=cfg.input_dim))

    def forward(self, inputs: Tensor) -> Tensor:
        cfg = self.config
        x = self.input_dropout(inputs)
        x += self.pos_emb(x)
        for i in range(cfg.num_layers):
            x = self.children[f"layer{i}"](x)
        x = self.output_norm(x)
        return x


class ResNetEncoder(BaseLayer):
    """The generic ResNet encoder."""

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
            None,
            "The normalization layer config.",
        )
        cfg.define("stage", ResNetStage.default_config(), "The stage config.")
        cfg.define(
            "num_blocks_per_stage",
            None,
            "A list of integers, representing number of blocks per stage.",
        )
        cfg.param_init = param_init.DefaultInitializer.default_config().set(
            # Equivalent to kaiming_normal_(mode='fan_out', nonlinearity='relu').
            fan="fan_out",
            distribution="normal",
            gain=math.sqrt(2),
        )
        cfg.dtype = jnp.float32
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        hidden_dim = cfg.hidden_dim
        # FIXME(ruoming): use weight standardization.
        # https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_resnet.py#L30-L40
        self._add_child(
            "conv1",
            Conv2D.default_config().set(
                window=(7, 7),
                padding=((3, 3), (3, 3)),
                strides=(2, 2),
                input_dim=3,
                output_dim=hidden_dim,
                bias=False,
                param_partition_spec=(None, None, None, "model"),
            ),
        )
        self._add_child("norm1", cfg.norm.set(dim=hidden_dim))
        for stage_i, num_blocks in enumerate(cfg.num_blocks_per_stage):
            output_dim = hidden_dim if stage_i == 0 else hidden_dim * 2
            self._add_child(
                f"stage{stage_i}",
                cfg.stage.set(
                    input_dim=hidden_dim,
                    output_dim=output_dim,
                    stride=1 if stage_i == 0 else 2,
                    num_blocks=num_blocks,
                ),
            )
            hidden_dim = output_dim
        assert hidden_dim == self.output_dim

    def forward(self, x: Tensor) -> Tensor:
        cfg = self.config
        x = self.conv1(x)
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
        return x


class Model(BaseLayer):
    """An encoder consisting of multiple transformer layers."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("encoder_2d", ResNetEncoder.default_config().set(norm=GroupNorm.default_config()), "The 2-D encoder.")
        cfg.define("encoder_1d", TransformerSequenceEncoder.default_config(), "The 1-D encoder.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("encoder_2d", cfg.encoder_2d)
        self._add_child("flatten", cfg.flatten)
        self._add_child("encoder_1d", cfg.encoder_1d)

    def forward(self, image: Tensor, label: Tensor) -> Tuple[Tensor, NestedTensor]:
        cfg = self.config
        x = self.conv_encoder(image)
        x = self.flatten(x)
        x = self.encoder(x)
        logits = self.fc(x)
        loss = - (jax.nn.log_softmax(logits) * jax.nn.one_hot(label, cfg.num_classes)).sum(axis=-1).mean()
        # [batch].
        predictions = jnp.argmax(logits, axis=-1)
        num_examples = float(label.shape[0])
        accuracy = jnp.equal(predictions, label).sum() / num_examples
        self.add_summary("loss", WeightedScalar(loss, num_examples))
        self.add_summary("accuracy", WeightedScalar(accuracy, num_examples))
        return loss, {'logits': logits}

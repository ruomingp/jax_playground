"""Vision transformer layers.

References:
- https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py
"""
import math
from typing import Any, Dict, Optional, Tuple

import jax.nn
from jax import numpy as jnp

import config as config_lib
import param_init
from attention import LearnedPositionalEmbedding, RepeatedTransformerLayer, StackedTransformerLayer
from config import config_for_class
from layers import Conv2D, Dropout, LayerNorm, Linear, set_dropout_rate_recursively
from metrics import WeightedScalar
from module import BaseLayer, Module, NestedTensor, ParameterSpec, Tensor
from param_init import DefaultInitializer, GaussianInitializer
from resnet import ResNetStage


def layer_norm_config(eps=1e-6):
    return LayerNorm.default_config().set(eps=eps)


class ResNetEncoder(BaseLayer):
    """The generic ResNet encoder.

    TODO(ruoming): move this to resnet.py.
    """

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
        cfg.param_init = DefaultInitializer.default_config().set(
            # Equivalent to kaiming_normal_(mode='fan_out', nonlinearity='relu').
            fan="fan_out",
            distribution="normal",
            scale=math.sqrt(2),
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

    @property
    def output_dim(self):
        cfg = self.config
        return cfg.hidden_dim * (2 ** (len(cfg.num_blocks_per_stage) - 1))

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


class ConvertToSequence(BaseLayer):
    """A layer to flatten 2-D images to 1-D sequences."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("patch_size", (16, 16), "The 2-D patch size.")
        cfg.define("input_dim", None, "The input feature dim.")
        cfg.define("output_dim", None, "The output feature dim.")
        cfg.define(
            "conv",
            Conv2D.default_config().set(
                # We may not need bias when we later apply a learnable positional embedding, but we use it here for
                # consistency with the original ViT implementation.
                bias=True,
                param_partition_spec=(None, None, None, "model"),
            ),
            "The convolution layer config.",
        )
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "conv",
            cfg.conv.set(
                window=cfg.patch_size,
                padding=((0, 0), (0, 0)),
                strides=cfg.patch_size,
                input_dim=cfg.input_dim,
                output_dim=cfg.output_dim,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Given an input tensor of shape [B, H, W, input_dim], converts to a tensor of shape [B, H*W, output_dim]."""
        x = self.conv(x)
        batch, height, width, output_dim = x.shape
        return jnp.reshape(x, (batch, height * width, output_dim))


class TransformerSequenceEncoder(BaseLayer):
    """A sequence encoder consisting of multiple transformer layers."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("input_dim", 0, "Input feature dim.")
        cfg.define("input_dropout", Dropout.default_config().set(rate=0.1), "Input dropout config.")
        cfg.define(
            "pos_emb", LearnedPositionalEmbedding.default_config(), "Positional embedding config."
        )
        # https://github.com/google-research/vision_transformer/blob/dc8ddbcdeefd281d6cc7fea0c97355495688ca9c/vit_jax/models.py#L189
        cfg.pos_emb.param_init = config_for_class(GaussianInitializer).set(std=0.02)
        cfg.define(
            "transformer",
            StackedTransformerLayer.default_config(),
            "The transformer layer stack config.",
        )
        # Vision transformer uses 'gelu' and dropout=0.1 by default.
        set_dropout_rate_recursively(cfg, dropout_rate=0.1)
        transformer_layer_cfg = cfg.transformer.layer
        transformer_layer_cfg.feed_forward.activation = "nn.gelu"
        transformer_layer_cfg.feed_forward.norm = layer_norm_config()
        transformer_layer_cfg.self_attention.norm = layer_norm_config()
        cfg.define(
            "global_feature_extraction",
            None,
            "Supported values are 'cls_token' or 'gap'. "
            "If 'cls_token', a CLS token will be prepended to the input 1-D sequence, "
            "whose output embedding will be used as the global feature. "
            "If 'gap', apply global average pooling on the output sequence of transformer layers.",
        )
        cfg.define(
            "output_norm",
            layer_norm_config(),
            "The normalization layer config for encoder output.",
        )
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("input_dropout", cfg.input_dropout)
        self._add_child("pos_emb", cfg.pos_emb.set(dim=cfg.input_dim))
        cfg.transformer.layer.input_dim = cfg.input_dim
        self._add_child("transformer", cfg.transformer)
        if cfg.global_feature_extraction not in ("cls_token", "gap"):
            raise NotImplementedError(
                f"Unsupported global_feature_extraction: {cfg.global_feature_extraction}. "
                "Must be 'cls_token' or 'gap'."
            )
        self._add_child("output_norm", cfg.output_norm.set(dim=cfg.input_dim))

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        param_specs = {}
        if cfg.global_feature_extraction == "cls_token":
            param_specs["cls_token"] = ParameterSpec(
                shape=(1, 1, cfg.input_dim),
                partition=(None, None, "model"),
                initializer=param_init.ConstantInitializer(0.0),
            )
        return param_specs

    def forward(self, inputs: Tensor) -> Tensor:
        """Given input sequences of shape [B, length, input_dim], computes global features of shape [B, input_dim]."""
        cfg = self.config
        batch_size, _, _ = inputs.shape
        if cfg.global_feature_extraction == "cls_token":
            inputs = jnp.concatenate(
                [self.parameters["cls_token"].tile((batch_size, 1, 1)), inputs], axis=1
            )
        x = self.input_dropout(inputs)
        x += self.pos_emb(x)
        x = self.transformer(x).data
        x = self.output_norm(x)
        if cfg.global_feature_extraction == "cls_token":
            x = x[:, 0, :]
        else:
            assert cfg.global_feature_extraction == "gap"
            x = jnp.mean(x, axis=1)
        return x


class Model(BaseLayer):
    """A Vision Transformer model for image classification."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("encoder_2d", None, "An optional 2-D encoder, e.g., ResNetEncoder.")
        cfg.define(
            "convert_to_sequence",
            ConvertToSequence.default_config(),
            "The layer to flatten 2-D images to 1-D sequences.",
        )
        cfg.define("encoder_1d", TransformerSequenceEncoder.default_config(), "The 1-D encoder.")
        cfg.define(
            "classifier",
            Linear.default_config().set(bias=True, param_partition_spec=("model", None)),
            "The layer to compute logits from global features.",
        )
        cfg.define(
            "num_classes",
            1000,
            "The number of classification classes.",
        )
        # https://github.com/google-research/vision_transformer/blob/e7d87a59784503b5bd8825ec368bca17822fb959/vit_jax/models.py#L76.
        cfg.param_init = param_init.DefaultInitializer.default_config().set(
            fan="fan_avg", distribution="uniform"
        )
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.encoder_2d is not None:
            self._add_child("encoder_2d", cfg.encoder_2d)
            feature_dim = self.encoder_2d.output_dim
        else:
            self.encoder_2d = lambda x: x  # identity
            feature_dim = 3
        self._add_child(
            "convert_to_sequence",
            cfg.convert_to_sequence.set(input_dim=feature_dim, output_dim=cfg.encoder_1d.input_dim),
        )
        self._add_child("encoder_1d", cfg.encoder_1d)
        self._add_child(
            "classifier",
            cfg.classifier.set(input_dim=cfg.encoder_1d.input_dim, output_dim=cfg.num_classes),
        )

    def forward(self, image: Tensor, label: Tensor) -> Tuple[Tensor, NestedTensor]:
        cfg = self.config
        self.vlog(
            3, "image=%s(%s) label=%s(%s)", image.dtype, image.shape, label.dtype, label.shape
        )
        x = self.encoder_2d(image.astype(self.dtype()))
        x = self.convert_to_sequence(x)
        x = self.encoder_1d(x)
        logits = self.classifier(x)
        loss = (
            -(jax.nn.log_softmax(logits) * jax.nn.one_hot(label, cfg.num_classes))
            .sum(axis=-1)
            .mean()
        )
        # [batch].
        predictions = jnp.argmax(logits, axis=-1)
        num_examples = float(label.shape[0])
        accuracy = jnp.equal(predictions, label).sum() / num_examples
        self.add_summary("loss", WeightedScalar(loss, num_examples))
        self.add_summary("accuracy", WeightedScalar(accuracy, num_examples))
        return loss, {"logits": logits}


_NAMED_VIT_MODELS = {
    "Test16": dict(num_layers=3, model_dim=8, num_heads=4),
    # Table 1 of https://arxiv.org/pdf/2010.11929.pdf.
    "B16": dict(num_layers=12, model_dim=768, num_heads=12),
    "B32": dict(num_layers=12, model_dim=768, num_heads=12, patch_size=(32, 32)),
    "L16": dict(num_layers=24, model_dim=1024, num_heads=16),
    "L32": dict(num_layers=24, model_dim=1024, num_heads=16, patch_size=(32, 32)),
    # When patch_size=(14, 14), use global average pooling for feature extraction so that the sequence length is 256
    # instead of 257.
    "H14": dict(
        num_layers=32,
        model_dim=1280,
        num_heads=16,
        patch_size=(14, 14),
        global_feature_extraction="gap",
        dtype=jnp.bfloat16,
    ),
    # Table 2 of https://arxiv.org/pdf/2106.04560.pdf.
    "G14": dict(
        num_layers=48,
        model_dim=1664,
        num_heads=16,
        feed_forward_dim=8192,
        patch_size=(14, 14),
        global_feature_extraction="gap",
        dtype=jnp.bfloat16,
        dropout_rate=0.0,
    ),
}


def _set_model_config(
    cfg,
    *,
    num_layers: int,
    model_dim: int,
    num_heads: int,
    feed_forward_dim: Optional[int] = None,
    image_size: Tuple[int, int] = (224, 224),
    patch_size: Tuple[int, int] = (16, 16),
    global_feature_extraction: str = "cls_token",
    dtype: jnp.dtype = jnp.float32,
    dropout_rate: float = 0.1,
    transformer_stack_cfg: Optional[config_lib.InstantiableConfig] = None,
):
    cfg.dtype = dtype
    if not all(i % p == 0 for i, p in zip(image_size, patch_size)):
        raise ValueError(f"patch_size ({patch_size}) must divide image_size ({image_size}) evenly")
    cfg.convert_to_sequence.patch_size = patch_size
    encoder_cfg = cfg.encoder_1d
    encoder_cfg.set(
        input_dim=model_dim,
        global_feature_extraction=global_feature_extraction,
    )
    if transformer_stack_cfg is not None:
        encoder_cfg.transformer = transformer_stack_cfg
    encoder_cfg.transformer.num_layers = num_layers
    seq_len = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]) + (
        1 if global_feature_extraction == "cls_token" else 0
    )
    encoder_cfg.pos_emb.shape = (seq_len,)
    feed_forward_dim = feed_forward_dim or model_dim * 4
    encoder_cfg.transformer.layer.feed_forward.hidden_dim = feed_forward_dim
    encoder_cfg.transformer.layer.self_attention.attention.num_heads = num_heads
    set_dropout_rate_recursively(cfg, dropout_rate)


def named_model_configs() -> config_lib.InstantiableConfig:
    models = dict(**_NAMED_VIT_MODELS)
    # G/14 with RepeatedTransformerLayer.
    models["G14-repeat"] = dict(
        **models["G14"],
        transformer_stack_cfg=RepeatedTransformerLayer.default_config().set(
            layer=TransformerSequenceEncoder.default_config().transformer.layer
        ),
    )

    config_map = {}
    for model_name, model_settings in models.items():
        cfg = Model.default_config()
        _set_model_config(cfg, **model_settings)
        config_map[model_name] = cfg
    return config_map

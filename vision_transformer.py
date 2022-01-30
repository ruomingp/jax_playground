"""Vision transformer layers.

References:
- https://github.com/google-research/vision_transformer/blob/main/vit_jax/models.py
"""
from attention import LearnedPositionalEmbedding, TransformerLayer
import config as config_lib
from module import BaseLayer
from layers import Dropout, LayerNorm
from param_init import GaussianInitializer


class Encoder(BaseLayer):
    """An encoder consisting of multiple transformer layers."""

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
            self._add_child(f"layer_{i:03d}", cfg.transformer.set(input_dim=cfg.input_dim))
        self._add_child("output_norm", cfg.output_norm.set(input_dim=cfg.input_dim))

    def forward(self, inputs):
        cfg = self.config
        x = self.input_dropout(inputs)
        x += self.pos_emb(x)
        for i in range(cfg.num_layers):
            x = self.children[f"layer_{i:03d}"](x)
        x = self.output_norm(x)
        return x

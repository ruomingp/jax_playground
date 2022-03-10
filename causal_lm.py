import math
from typing import Dict, Optional, Tuple

import jax
from jax import numpy as jnp
from jax.experimental.pjit import PartitionSpec

import config as config_lib
from attention import (
    LearnedPositionalEmbedding,
    PipelinedTransformerLayer,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerLayer,
    make_causal_mask,
)
from layers import Dropout, Embedding, LayerNorm, set_dropout_rate_recursively
from metrics import WeightedScalar
from module import BaseLayer, Module, NestedPartitionSpec, NestedTensor, ParameterSpec, Tensor
from param_init import DefaultInitializer
from utils import with_sharding_constraint


class LmHead(BaseLayer):
    """LM head layer."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("vocab_size", 0, "Size of the LM vocabulary.")
        cfg.define("embedding_dim", 0, "Dimensionality of vocabulary embedding table.")
        cfg.param_partition_spec = (None, "model")
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        return dict(
            weight=ParameterSpec(
                shape=(cfg.vocab_size, cfg.embedding_dim),
                partition=cfg.param_partition_spec,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return jnp.einsum("bsh,vh->bsv", x, self.parameters["weight"])


class Model(BaseLayer):
    """Autoregressive decoder-only transformer sequence model."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("vocab_size", 0, "Size of vocabulary.")
        cfg.define("source_length", 0, "Maximum input sequence length.")
        cfg.define(
            "hidden_dim", 0, "Dimensionality of embeddings and inputs to each transfomer layer."
        )
        cfg.define("dropout_rate", 0.0, "Dropout rate applied throughout model.")
        cfg.define("emb", Embedding.default_config(), "Vector from input-ID lookup table.")
        cfg.define("pos_emb", LearnedPositionalEmbedding.default_config(), "Positional embeddings.")
        cfg.define(
            "transformer", StackedTransformerLayer.default_config(), "Transformer model trunk."
        )
        cfg.define("output_norm", layer_norm_config(), "Layer norm applied to transformer output.")
        cfg.define("mask_input_id", 0, "Int ID of the inputs to be masked for self-attention.")
        cfg.define(
            "lm_head",
            None,
            "Optional LmHead layer maps the hidden state to vocab logits (if None use embedding).",
        )
        cfg.param_init = DefaultInitializer.default_config().set(
            fan=None, scale=0.02, distribution="normal"
        )
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        set_dropout_rate_recursively(cfg, dropout_rate=cfg.dropout_rate)
        self._add_child("embedding_dropout", Dropout.default_config().set(rate=cfg.dropout_rate))
        self._add_child("emb", cfg.emb.set(dim=cfg.hidden_dim, num_embeddings=cfg.vocab_size))
        self._add_child("pos_emb", cfg.pos_emb.set(dim=cfg.hidden_dim, shape=(cfg.source_length,)))
        cfg.transformer.layer.input_dim = cfg.hidden_dim
        self._add_child("transformer", cfg.transformer)
        self._add_child("output_norm", cfg.output_norm.set(dim=cfg.hidden_dim))
        self._add_child("output_dropout", Dropout.default_config().set(rate=cfg.dropout_rate))
        if cfg.lm_head is not None:
            self._add_child(
                "lm_head", cfg.lm_head.set(vocab_size=cfg.vocab_size, embedding_dim=cfg.hidden_dim)
            )

    def create_partition_specs_recursively(self) -> NestedPartitionSpec:
        # TODO(ruoming_pang): Check if there is a better work around for tied param init/specs.
        specs = super().create_partition_specs_recursively()
        if self.config.lm_head is None:
            specs["emb_attend"] = {}
        return dict(sorted(specs.items()))

    def initialize_parameters_recursively(self, prng_key: jax.random.KeyArray) -> NestedTensor:
        params = super().initialize_parameters_recursively(prng_key)
        if self.config.lm_head is None:
            params["emb_attend"] = {}
        return params

    def forward(
        self, inputs: Tensor, targets: Optional[Tensor] = None, return_aux: bool = False
    ) -> Tuple[Tensor, NestedTensor]:
        self.vlog(3, "image=%s(%s)", inputs.dtype, inputs.shape)
        if targets is not None:
            self.vlog(3, "label=%s(%s)", targets.dtype, targets.shape)
        # [batch, source_length]
        x = self.emb(inputs)
        # [batch, source_length, self.config.hidden_dim]
        x = x + self.pos_emb(x)
        x = self.embedding_dropout(x)
        x = self.transformer(x, self_attention_mask=self._attention_mask(inputs)).data
        x = self.output_norm(x)
        x = self.output_dropout(x)
        if self.config.lm_head is None:
            logits = self.emb(
                x,
                method="attend",
                context=self.get_invocation_context().add_child(
                    "emb", output_collection_name="emb_attend"
                ),
            )
        else:
            logits = self.lm_head(x)
        if logits.dtype in (jnp.bfloat16, jnp.float16):
            logits = logits.astype(jnp.float32)
        logits = with_sharding_constraint(logits, PartitionSpec("data", None, "model"))
        # [batch source_length, vocab_size]
        loss = None
        if targets is not None:
            loss = self._loss(inputs, logits, targets)
        aux_outputs = {}
        if return_aux:
            # Return the logits and output pre LM head (useful for downstream tasks).
            #
            # N.B. Do not enable for large-scale training since auxiliary outputs are not partitioned.
            # TODO(rpang): support partitioning of auxiliary outputs.
            aux_outputs["logits"] = logits
            aux_outputs["hidden_state"] = x
        return loss, aux_outputs

    def _loss(self, inputs: Tensor, logits: Tensor, targets: Tensor) -> Tensor:
        live_inputs = inputs != self.config.mask_input_id
        num_inputs = live_inputs.sum().astype(jnp.float32)
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), targets).sum() / num_inputs
        self.add_summary("accuracy", WeightedScalar(accuracy, num_inputs))
        # TODO(tom_gunter): Implement a more stable cross entropy loss.
        loss = -(jax.nn.log_softmax(logits) * jax.nn.one_hot(targets, self.config.vocab_size)).sum(
            axis=-1
        )
        loss = (loss * live_inputs).sum() / num_inputs
        self.add_summary("loss", WeightedScalar(loss, num_inputs))
        self.add_summary("perplexity", WeightedScalar(jnp.exp(loss), num_inputs))
        return loss

    def _attention_mask(self, inputs: Tensor) -> Tensor:
        causal_mask = make_causal_mask(inputs.shape[-1])
        inputs_mask = inputs == self.config.mask_input_id
        return causal_mask[None, ...] | inputs_mask[:, None, :]


def layer_norm_config(eps=1e-5):
    return LayerNorm.default_config().set(eps=eps)


def gpt2_transformer_cfg(
    transformer_cls: BaseLayer, *, num_layers: int, hidden_dim: int, num_heads: int
):
    """Build an autoregressive transformer decoder config in the style of 'GPT2':
    <http://www.persagen.com/files/misc/radford2019language.pdf>

    Args:
        transformer_cls: {StackedTransformerLayer, RepeatedTransformerLayer, PipelinedTransformerLayer}
        num_layers: Number of transformer decoder layers.
        hidden_dim: Dimension of embeddings and input/output to each transformer layer.
        num_heads: Number of attention heads per transformer layer.
    Returns:
        transformer_cls config.
    """
    assert transformer_cls in [
        StackedTransformerLayer,
        RepeatedTransformerLayer,
        PipelinedTransformerLayer,
    ]

    # TransformerLayer.
    layer_cfg = TransformerLayer.default_config()
    # Feed-forward transformer layer config.
    layer_cfg.feed_forward.activation = "nn.gelu"
    layer_cfg.feed_forward.norm = layer_norm_config()
    layer_cfg.feed_forward.input_dim = hidden_dim
    layer_cfg.feed_forward.hidden_dim = 4 * hidden_dim
    layer_cfg.feed_forward.structure = "prenorm"
    # Self attention transformer layer config.
    layer_cfg.self_attention.norm = layer_norm_config()
    layer_cfg.self_attention.attention.num_heads = num_heads
    layer_cfg.self_attention.structure = "prenorm"

    # Initialization.
    def default_gpt2_initializer_cfg(scale=0.02):
        return DefaultInitializer.default_config().set(fan=None, distribution="normal", scale=scale)

    def residual_gpt2_initializer_cfg(scale=0.02):
        # Section 2.3: "Scale weights on residual path by 1/sqrt(num_layers)".
        scale = scale / math.sqrt(2 * num_layers)  # 2 x residuals per layer.
        return DefaultInitializer.default_config().set(fan=None, distribution="normal", scale=scale)

    layer_cfg.feed_forward.linear1.param_init = default_gpt2_initializer_cfg()
    layer_cfg.feed_forward.linear2.param_init = residual_gpt2_initializer_cfg()
    layer_cfg.self_attention.attention.input_linear.param_init = default_gpt2_initializer_cfg()
    layer_cfg.self_attention.attention.output_linear.param_init = residual_gpt2_initializer_cfg()

    return transformer_cls.default_config().set(num_layers=num_layers, layer=layer_cfg)

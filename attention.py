import math
from dataclasses import dataclass
from typing import Optional

import jax
from absl import logging
from jax import numpy as jnp

import config as config_lib
import param_init
from layers import check_numerics, LayerNorm, Dropout, Linear, get_activation_fn
from module import Module, NestedParameters

Tensor = jnp.ndarray


def make_causal_mask(seq_len: int) -> Tensor:
    """Generates attention mask for causal masking.

    Args:
        seq_len: sequence length.

    Returns:
        A boolean tensor of shape [seq_len, seq_len] where the value at [i, j] = False if i >= j.
    """
    indexes = jnp.arange(seq_len)
    return indexes[:, None] < indexes[None, :]


def make_segment_mask(*, source_segments: Tensor, target_segments: Tensor) -> Tensor:
    """Generates attention mask given the segment ids.

    ... such that positions belonging to different segments cannot attend to each other.

    Args:
        source_segments: An integer tensor of shape [batch..., source_length].
        target_segments: An integer tensor of shape [batch..., target_length].

    Returns:
        A boolean tensor of shape [batch..., target_length, source_length] where the value at [..., i, j] = False if
        target_segments[..., i] == source_segments[..., j].
    """
    target_segments = jnp.expand_dims(target_segments, -1)
    source_segments = jnp.expand_dims(source_segments, -2)
    return source_segments != target_segments


def t5_relative_position_bucket(
    relative_position, *, bidirectional=True, num_buckets=32, max_distance=128
):
    """Computes relative position buckets with the T5 algorithm.

    Based on HuggingFace code:
    https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/t5/modeling_t5.py#L346-L392

    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
    small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
    positions >= max_distance map to the same bucket. All relative positions <= -max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on.

    Args:
        relative_position: an int32 Tensor of any shape.
        bidirectional: a boolean - whether the attention is bidirectional.
        num_buckets: an integer.
        max_distance: an integer.

    Returns:
        A Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets).
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).astype(jnp.int32) * num_buckets
        relative_position = jnp.abs(relative_position)
    else:
        relative_position = -jnp.minimum(
            relative_position, jnp.zeros_like(relative_position)
        )
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = max_exact + (
        jnp.log(relative_position.astype(jnp.float32) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(jnp.int32)
    relative_position_if_large = jnp.minimum(
        # relative_position_if_large, jnp.full_like(relative_position_if_large, num_buckets - 1)
        relative_position_if_large,
        num_buckets - 1,
    )

    relative_buckets += jnp.where(
        is_small, relative_position, relative_position_if_large
    )
    return relative_buckets


class MultiheadLinearInit(param_init.DefaultInit):
    """Initialization settings for 3-D projection weights used by MultiheadAttention.

    The default fan-in/fan-out calculation does not work for 3-D weights of shape [model_dim, num_heads, per_head_dim],
    so we need a custom initialization setting here.
    """

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("type", None, '"input" or "output"')
        return cfg

    def calculate_fan_in_and_fan_out(self, name: str, shape: param_init.Shape):
        cfg = self.config
        if len(shape) != 3:
            raise ValueError(f"Unexpected parameter shape {shape}")
        model_dim, num_heads, per_head_dim = tuple(shape)
        if cfg.type == "input":
            return model_dim, num_heads * per_head_dim
        elif cfg.type == "output":
            return num_heads * per_head_dim, model_dim
        else:
            raise NotImplementedError(f"Unknown linear type ({cfg.type})")


class _BaseMultiheadLinear(Module):
    """The linear layer used for multi-head attention.

    It uses einsum for efficient computation on TPU to avoid reshaping.
    """

    @classmethod
    def _config_for_linear_type(cls, linear_type: str):
        cfg = super().default_config()
        cfg.define("model_dim", 0, "Feature dim.")
        cfg.define(
            "num_heads", 0, "Number of attention heads. Must divide hidden_dim evenly."
        )
        cfg.define("per_head_dim", 0, "Dimension per head.")
        cfg.define("bias", True, "Whether the linear modules have biases.")
        assert linear_type in ("input", "output")
        cfg.param_init = MultiheadLinearInit.default_config().set(type=linear_type)
        return cfg

    def _initialize_module_parameters(
        self, *, prng_key: jax.random.KeyArray
    ) -> NestedParameters:
        cfg = self.config
        params = dict(
            weight=self._initialize_parameter(
                "weight",
                prng_key=prng_key,
                shape=(cfg.model_dim, cfg.num_heads, cfg.per_head_dim),
            )
        )
        if cfg.bias:
            params["bias"] = jnp.zeros(shape=self._bias_shape, dtype=self.dtype())
        return params

    def forward(self, inputs: jnp.ndarray) -> jnp.ndarray:
        params = self.parameters
        outputs = jnp.einsum(self._einsum_expr, inputs, params["weight"])
        return outputs + params.get("bias", 0)


class MultiheadInputLinear(_BaseMultiheadLinear):
    @classmethod
    def default_config(cls):
        cfg = super()._config_for_linear_type(linear_type="input")
        return cfg

    @property
    def _einsum_expr(self):
        return "btd,dnh->btnh"

    @property
    def _bias_shape(self):
        cfg = self.config
        return cfg.num_heads, cfg.per_head_dim


class MultiheadOutputLinear(_BaseMultiheadLinear):
    @classmethod
    def default_config(cls):
        cfg = super()._config_for_linear_type(linear_type="output")
        return cfg

    @property
    def _einsum_expr(self):
        return "btnh,dnh->btd"

    @property
    def _bias_shape(self):
        cfg = self.config
        return (cfg.model_dim,)


def masked_softmax(logits: Tensor, mask: Optional[Tensor] = None):
    """Computes softmax with optional masking.

    Args:
        logits: a Tensor of any shape.
        mask: a mask Tensor that is broadcastable with logits. It can be a boolean tensor, where each True value
              represents that attention is masked for the corresponding position pair, or a float tensor, which will
              be added to the attention logits (therefore a -inf represents a masked logit).

    Returns:
        A Tensor of same shape as logits.
    """
    check_numerics(logits)
    if mask is not None:
        if mask.dtype == jnp.bool_:
            min_value = jnp.finfo(logits.dtype).min
            mask = mask.astype(logits.dtype) * 0.5 * min_value
        else:
            assert jnp.issubdtype(
                mask.dtype, jnp.floating
            ), f"Expected float tensor for mask, got {mask}"
        logits = logits + mask
    if logits.dtype in (jnp.bfloat16, jnp.float16):
        # Avoid computing softmax in 16-bit floats.
        logits = logits.astype(jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)
    check_numerics(probs)
    return probs


class MultiheadAttention(Module):
    """A basic multi-head attention layer.

    Differences from jnp.nn.MultiheadAttention:
    - Use of einsum for efficient computation on TPU to avoid reshaping;
    - Separate weights for {q,k,v}_proj for proper weight initialization that depends on fan-out and efficient TPU
      execution (where split is not free).
    """

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("query_dim", 0, "Input query feature dim.")
        cfg.define("key_dim", 0, "Input key feature dim.")
        cfg.define("value_dim", 0, "Input value feature dim.")
        cfg.define("output_dim", None, "Output feature dim. If None, use query_dim.")
        cfg.define("hidden_dim", None, "Hidden feature dim. If None, use query_dim.")
        cfg.define(
            "num_heads", 0, "Number of attention heads. Must divide hidden_dim evenly."
        )
        cfg.define(
            "input_linear",
            MultiheadInputLinear.default_config(),
            "Config used for the Q,K,V projections.",
        )
        cfg.define(
            "output_linear",
            MultiheadOutputLinear.default_config(),
            "Config used for the output projection.",
        )
        cfg.define("dropout", Dropout.default_config(), "The dropout layer.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        for name, dim in (
            ("q", cfg.query_dim),
            ("k", cfg.key_dim),
            ("v", cfg.value_dim),
            ("o", self.output_dim()),
        ):
            proj_cfg = cfg.output_linear if name == "o" else cfg.input_linear
            proj_cfg.model_dim = dim
            proj_cfg.num_heads = cfg.num_heads
            proj_cfg.per_head_dim = self.per_head_dim()
            self._add_child(f"{name}_proj", proj_cfg)
        self._add_child("dropout", cfg.dropout)

    def output_dim(self):
        cfg = self.config
        return cfg.output_dim or cfg.query_dim

    def per_head_dim(self):
        cfg = self.config
        hidden_dim = cfg.hidden_dim or cfg.query_dim
        if hidden_dim % cfg.num_heads != 0:
            raise ValueError(
                f"num_heads ({cfg.num_heads}) must divide hidden_dim ({hidden_dim})"
            )
        return hidden_dim // cfg.num_heads

    @dataclass
    class Output:
        # [batch, target_length, output_dim]. The attention output.
        data: Tensor
        # [batch, num_heads, target_length, source_length]. The attention probabilities.
        probs: Tensor

    def forward(
        self,
        query: Tensor,
        *,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Output:
        """Computes attention for the given query, key, value, and mask.

        Args:
            query: a Tensor of shape [batch, target_length, target_dim].
            key:   a Tensor of shape [batch, source_length, source_dim].
            value: a Tensor of shape [batch, source_length, source_dim].
            mask:  a mask Tensor of shape [batch, target_length, source_length] or [batch, num_heads, target_length,
                source_length]. It can be a boolean tensor, where each True value represents that attention is masked
                for the corresponding position pair, or a float tensor, which will be added to the attention logits
                (therefore a -inf represents a masked position pair).

        Returns:
            An Output instance, where .data is of the same shape as query and .probs is of shape
            [batch, num_heads, target_length, source_length].
        """
        q_scale = self.per_head_dim() ** -0.5
        q_proj = self.q_proj(query) * q_scale
        k_proj = self.k_proj(key)
        v_proj = self.v_proj(value)
        logits = jnp.einsum("btnh,bsnh->bnts", q_proj, k_proj)
        # logging.info("MultiheadAttention.logits=%s", logits[0, 0, 0].reshape([-1]))
        if mask is not None and mask.ndim == 3:
            # [batch, 1, target_length, source_length].
            mask = mask[:, None, :, :]
        probs = masked_softmax(logits, mask=mask)
        probs = self.dropout(probs)
        context = jnp.einsum("bnts,bsnh->btnh", probs, v_proj)
        logging.info("MultiheadAttention.context=%s", context[0, 0].reshape([-1]))
        # [batch, target_length, output_dim].
        outputs = self.o_proj(context)
        return self.Output(data=outputs, probs=probs)


class TransformerAttentionLayer(Module):
    """A Transformer attention layer that can be used for either self-attention or cross-attention."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("target_dim", 0, "Input target feature dim.")
        cfg.define("source_dim", 0, "Input source feature dim.")
        cfg.define(
            "norm", LayerNorm.default_config(), "The normalization layer config."
        )
        cfg.define(
            "attention",
            MultiheadAttention.default_config(),
            "The attention layer config.",
        )
        cfg.define("dropout", Dropout.default_config(), "The dropout layer config.")
        cfg.define(
            "structure",
            "prenorm",
            "The inner structure of the layer: prenorm or postnorm. "
            "See https://arxiv.org/abs/2002.04745 for background.",
        )
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("norm", cfg.norm.set(dim=cfg.target_dim))
        self._add_child(
            "attention",
            cfg.attention.set(
                query_dim=cfg.target_dim,
                key_dim=cfg.source_dim,
                value_dim=cfg.source_dim,
                output_dim=cfg.target_dim,
            ),
        )
        self._add_child("dropout", cfg.dropout)

    @dataclass
    class Output:
        # [batch, target_length, output_dim]. The attention output.
        data: Tensor
        # The attention probabilities returned by the attention layer.
        probs: Tensor

    def forward(
        self,
        *,
        target: Tensor,
        source: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ):
        """Computes attention with target as query and source as key and value.

        Args:
            target: a Tensor of shape [batch, target_length, target_dim].
            source: a Tensor of shape [batch, source_length, source_dim].
                If None, uses norm(target) as source (self-attention)
            mask: a mask Tensor of shape [batch, target_length, source_length].

        Returns:
            An Output instance, where .data is of the same shape as target and .probs is of shape
            [batch, num_heads, target_length, source_length].
        """
        cfg = self.config
        if cfg.structure == "prenorm":
            skip_input = target  # pre-norm: where normalization happens within the residual part.
            norm_target = self.norm(target)
            if source is None:
                source = norm_target  # self attention
            atten_output = self.attention(
                query=norm_target, key=source, value=source, mask=mask
            )
            data = skip_input + self.dropout(atten_output.data)
        elif cfg.structure == "postnorm":
            # This is the structure used by the original Transformer, BERT, and RoBERTa.
            if source is None:
                source = target  # self attention
            atten_output = self.attention(
                query=target, key=source, value=source, mask=mask
            )
            logging.info("atten_output=%s", atten_output.data[0, 0])
            # Post-norm: norm applied on the sum of input and attention output.
            data = self.norm(target + self.dropout(atten_output.data))
        else:
            raise NotImplementedError(cfg.structure)
        return self.Output(data=data, probs=atten_output.probs)


class TransformerFeedForwardLayer(Module):
    """A Transformer feed-forward layer."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("input_dim", 0, "Input feature dim.")
        cfg.define("hidden_dim", 0, "The hidden dim.")
        cfg.define(
            "linear_tpl",
            Linear.default_config(),
            "Whether the linear modules have biases.",
        )
        cfg.define("norm", LayerNorm.default_config(), "The normalization layer config.")
        cfg.define("activation", "nn.relu", "The activation function.")
        cfg.define("dropout", Dropout.default_config(), "The dropout layer config.")
        cfg.define(
            "structure",
            "prenorm",
            "The inner structure of the layer: prenorm or postnorm. "
            "See https://arxiv.org/abs/2002.04745 for background.",
        )
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("norm", cfg.norm.set(dim=cfg.input_dim))
        self._add_child(
            "linear1",
            cfg.linear_tpl.set(input_dim=cfg.input_dim, output_dim=cfg.hidden_dim),
        )
        self._add_child(
            "linear2",
            cfg.linear_tpl.set(input_dim=cfg.hidden_dim, output_dim=cfg.input_dim),
        )
        self._add_child("dropout1", cfg.dropout)
        self._add_child("dropout2", cfg.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        cfg = self.config
        if cfg.structure == "prenorm":
            x = self.norm(inputs)
            x = self.linear1(x)
            x = get_activation_fn(cfg.activation)(x)
            x = self.dropout1(x)
            x = self.linear2(x)
            x = self.dropout2(x)
            x += inputs
        elif cfg.structure == "postnorm":
            x = self.linear1(inputs)
            x = get_activation_fn(cfg.activation)(x)
            x = self.linear2(x)
            x = self.dropout1(x)
            x = self.norm(x + inputs)
        else:
            raise NotImplementedError(cfg.structure)
        return x


class TransformerLayer(Module):
    """A Transformer layer.

    Unlike jnp.nn.TransformerLayer, this allows components to be customized, e.g., replacing vanilla attention with
    relative positional attention from TransformerXL/DeBERTa or replacing feed-forward with a mixture-of-expert
    feed-forward layer.
    """

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("input_dim", 0, "Input feature dim.")
        cfg.define(
            "self_attention",
            TransformerAttentionLayer.default_config(),
            "The self-attention layer config.",
        )
        cfg.define(
            "cross_attention", None, "If not None, the cross-attention layer config."
        )
        cfg.define(
            "feed_forward",
            TransformerFeedForwardLayer.default_config(),
            "The feed-forward layer config.",
        )
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "self_attention",
            cfg.self_attention.set(target_dim=cfg.input_dim, source_dim=cfg.input_dim),
        )
        self._add_child("feed_forward", cfg.feed_forward.set(input_dim=cfg.input_dim))
        if cfg.cross_attention is not None:
            self._add_child(
                "cross_attention", cfg.cross_attention.set(target_dim=cfg.input_dim)
            )

    @dataclass
    class Output:
        # [batch, target_length, output_dim]. The attention output.
        data: Tensor
        # The attention probabilities returned by the self-attention layer.
        self_attention_probs: Tensor
        # The attention probabilities returned by the cross-attention layer.
        cross_attention_probs: Optional[Tensor]

    def forward(
        self,
        data: Tensor,
        *,
        self_attention_mask: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Output:
        self_atten_outputs = self.self_attention(target=data, mask=self_attention_mask)
        data = self_atten_outputs.data
        if cross_attention_data is not None:
            cross_atten_outputs = self.cross_attention(
                target=data, source=cross_attention_data, mask=cross_attention_mask
            )
            data = cross_atten_outputs.data
            cross_attention_probs = cross_atten_outputs.probs
        else:
            cross_attention_probs = None
        data = self.feed_forward(data)
        return self.Output(
            data=data,
            self_attention_probs=self_atten_outputs.probs,
            cross_attention_probs=cross_attention_probs,
        )

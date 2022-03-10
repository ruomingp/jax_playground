"""Attention layers with pjit partition specs.

On mask tensors:
A mask tensor can have shape [batch, target_length, source_length] or [batch, num_heads, target_length, source_length].
It can be a boolean tensor, where each True value represents that attention is masked for the corresponding position
pair, or a float tensor, which will be added to the attention logits (therefore a -inf represents a masked position
pair). mask=None represents an all-zero mask---no position pair is masked.
"""

import math
from typing import Dict, NamedTuple, Optional

import jax
from jax import numpy as jnp

import config as config_lib
import param_init
from layers import Dropout, LayerNorm, Linear, get_activation_fn
from module import BaseLayer, FactorizationSpec, Module, ParameterSpec
from pipeline import Pipeline
from repeat import Repeat
from utils import Tensor, check_numerics, shapes


def make_causal_mask(seq_len: int) -> Tensor:
    """Generates attention mask for causal masking.

    Args:
        seq_len: sequence length.

    Returns:
        A boolean tensor of shape [seq_len, seq_len] where the value at [i, j] = False if i >= j.
    """
    indexes = jnp.arange(seq_len)
    return jax.lax.lt(indexes[:, None], indexes[None, :])


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
    return jax.lax.ne(source_segments, target_segments)


class LearnedPositionalEmbedding(BaseLayer):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("dim", 0, "Input feature dim.")
        cfg.define("shape", (None,), "The sequence shape.")
        cfg.param_partition_spec = (None, None, "model")
        # By default, initialize to Gaussian with std=1/sqrt(dim), e.g., 0.036 when dim=768.
        #
        # This is the same as:
        # https://github.com/pytorch/fairseq/blob/master/fairseq/modules/positional_embedding.py#L26
        #
        # BERT uses std=0.02 regardless of dim:
        # https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/modeling.py#L492-L495
        cfg.param_init = param_init.DefaultInitializer.default_config().set(
            fan="fan_out", distribution="normal"
        )
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        return dict(
            weight=ParameterSpec(
                shape=[1] + list(cfg.shape) + [cfg.dim],
                partition=cfg.param_partition_spec,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        emb = self.parameters["weight"]
        assert x.shape[1:] == emb.shape[1:], f"Invalid input shape: {x.shape} vs. {emb.shape}"
        return emb


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
        relative_position = -jnp.minimum(relative_position, jnp.zeros_like(relative_position))
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

    relative_buckets += jnp.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


class MultiheadLinearInit(param_init.DefaultInitializer):
    """Initialization settings for 3-D projection weights used by MultiheadAttention.

    The default fan-in/fan-out calculation does not work for 3-D weights of shape [model_dim, num_heads, per_head_dim],
    so we need a custom initialization setting here.
    """

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("type", None, '"input" or "output"')
        return cfg

    def _compute_fan_axes(self, name: str, shape: param_init.Shape) -> param_init.FanAxes:
        cfg = self.config
        if len(shape) != 3:
            # model_dim, num_heads, per_head_dim = tuple(shape)
            raise ValueError(f"Unexpected parameter shape {shape}")
        if cfg.type == "input":
            return param_init.FanAxes(in_axis=0, out_axis=(1, 2))
        elif cfg.type == "output":
            return param_init.FanAxes(in_axis=(1, 2), out_axis=0)
        else:
            raise NotImplementedError(f"Unknown linear type ({cfg.type})")


class _BaseMultiheadLinear(BaseLayer):
    """The linear layer used for multi-head attention.

    It uses einsum for efficient computation on TPU to avoid reshaping.
    """

    @classmethod
    def _config_for_linear_type(cls, linear_type: str):
        cfg = super().default_config()
        cfg.define("model_dim", 0, "Feature dim.")
        cfg.define("num_heads", 0, "Number of attention heads. Must divide hidden_dim evenly.")
        cfg.define("per_head_dim", 0, "Dimension per head.")
        cfg.define("bias", True, "Whether the linear modules have biases.")
        # Shard the 'num_heads' axis by the 'model' dim of the mesh.
        cfg.param_partition_spec = (None, "model", None)
        assert linear_type in ("input", "output")
        cfg.param_init = MultiheadLinearInit.default_config().set(type=linear_type)
        return cfg

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        cfg = self.config
        params = dict(
            weight=ParameterSpec(
                shape=(cfg.model_dim, cfg.num_heads, cfg.per_head_dim),
                partition=cfg.param_partition_spec,
                factorization=FactorizationSpec(axes=("row", None, "col")),
            )
        )
        if cfg.bias:
            params["bias"] = self._bias_spec
        return params

    def forward(self, inputs: Tensor) -> Tensor:
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
    def _bias_spec(self):
        cfg = self.config
        return ParameterSpec(
            shape=(cfg.num_heads, cfg.per_head_dim),
            partition=cfg.param_partition_spec[-2:],
        )


class MultiheadOutputLinear(_BaseMultiheadLinear):
    @classmethod
    def default_config(cls):
        cfg = super()._config_for_linear_type(linear_type="output")
        return cfg

    @property
    def _einsum_expr(self):
        return "btnh,dnh->btd"

    @property
    def _bias_spec(self):
        cfg = self.config
        return ParameterSpec(
            shape=(cfg.model_dim,),
            partition=cfg.param_partition_spec[:1],
        )


def masked_softmax(logits: Tensor, mask: Optional[Tensor] = None):
    """Computes softmax with optional masking.

    Args:
        logits: a Tensor of any shape.
        mask: a mask Tensor that is broadcastable with logits. See ``On mask tensors`` in the file comments.

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


class MultiheadAttention(BaseLayer):
    """A basic multi-head attention layer.

    Differences from torch.nn.MultiheadAttention:
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
        cfg.define("num_heads", 0, "Number of attention heads. Must divide hidden_dim evenly.")
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
            raise ValueError(f"num_heads ({cfg.num_heads}) must divide hidden_dim ({hidden_dim})")
        return hidden_dim // cfg.num_heads

    class Output(NamedTuple):
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
            mask:  See ``On mask tensors`` in the file comments.

        Returns:
            An Output instance, where .data is of the same shape as query and .probs is of shape
            [batch, num_heads, target_length, source_length].
        """
        q_scale = self.per_head_dim() ** -0.5
        q_proj = self.q_proj(query) * q_scale
        k_proj = self.k_proj(key)
        v_proj = self.v_proj(value)
        logits = jnp.einsum("btnh,bsnh->bnts", q_proj, k_proj)
        if mask is not None and mask.ndim == 3:
            # [batch, 1, target_length, source_length].
            mask = mask[:, None, :, :]
        probs = masked_softmax(logits, mask=mask)
        probs = self.dropout(probs)
        context = jnp.einsum("bnts,bsnh->btnh", probs, v_proj).astype(v_proj.dtype)
        # [batch, target_length, output_dim].
        outputs = self.o_proj(context)
        return self.Output(data=outputs, probs=probs[0])


class TransformerAttentionLayer(BaseLayer):
    """A Transformer attention layer that can be used for either self-attention or cross-attention."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("target_dim", 0, "Input target feature dim.")
        cfg.define("source_dim", 0, "Input source feature dim.")
        cfg.define("norm", LayerNorm.default_config(), "The normalization layer config.")
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

    class Output(NamedTuple):
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
            mask: See ``On mask tensors`` in the file comments.

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
            atten_output = self.attention(query=norm_target, key=source, value=source, mask=mask)
            data = skip_input + self.dropout(atten_output.data)
        elif cfg.structure == "postnorm":
            # This is the structure used by the original Transformer, BERT, and RoBERTa.
            if source is None:
                source = target  # self attention
            atten_output = self.attention(query=target, key=source, value=source, mask=mask)
            # Post-norm: norm applied on the sum of input and attention output.
            data = self.norm(target + self.dropout(atten_output.data))
        else:
            raise NotImplementedError(cfg.structure)
        return self.Output(data=data, probs=atten_output.probs)


class TransformerFeedForwardLayer(BaseLayer):
    """A Transformer feed-forward layer."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("input_dim", 0, "Input feature dim.")
        cfg.define("hidden_dim", 0, "The hidden dim.")
        cfg.define(
            "linear1",
            Linear.default_config().set(param_partition_spec=[None, "model"]),
            "Config for the first linear layer.",
        )
        cfg.define(
            "linear2",
            Linear.default_config().set(param_partition_spec=["model", None]),
            "Config for the second linear layer.",
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
            cfg.linear1.set(input_dim=cfg.input_dim, output_dim=cfg.hidden_dim),
        )
        self._add_child(
            "linear2",
            cfg.linear2.set(input_dim=cfg.hidden_dim, output_dim=cfg.input_dim),
        )
        if cfg.structure == "prenorm":
            self._add_child("dropout1", cfg.dropout)
            self._add_child("dropout2", cfg.dropout)
        else:
            self._add_child("dropout", cfg.dropout)

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
            x = self.dropout(x)
            x = self.norm(x + inputs)
        else:
            raise NotImplementedError(cfg.structure)
        return x


class TransformerLayer(BaseLayer):
    """A Transformer layer.

    Unlike torch.nn.TransformerLayer, this allows components to be customized, e.g., replacing vanilla attention with
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
        cfg.define("cross_attention", None, "If not None, the cross-attention layer config.")
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
            self._add_child("cross_attention", cfg.cross_attention.set(target_dim=cfg.input_dim))

    class Output(NamedTuple):
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


class StackedTransformerLayer(BaseLayer):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("num_layers", None, "The number of layers in the stack.")
        cfg.define(
            "layer", TransformerLayer.default_config(), "Config for each layer in the stack."
        )
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._layers = []
        for i in range(cfg.num_layers):
            self._layers.append(self._add_child(f"layer{i}", cfg.layer))

    def forward(
        self,
        data: Tensor,
        *,
        self_attention_mask: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> TransformerLayer.Output:
        all_layer_outputs = []
        for layer in self._layers:
            layer_outputs: TransformerLayer.Output = layer(
                data,
                self_attention_mask=self_attention_mask,
                cross_attention_data=cross_attention_data,
                cross_attention_mask=cross_attention_mask,
            )
            all_layer_outputs.append(layer_outputs)
            data = layer_outputs.data
        aux_outputs = {}
        for field in TransformerLayer.Output._fields:
            if field == "data":
                continue
            values = [getattr(output, field) for output in all_layer_outputs]
            if any(v is None for v in values):
                assert all(v is None for v in values), f"{field}: {values}"
                aux_outputs[field] = None
            else:
                aux_outputs[field] = jnp.stack(values, axis=0)

        return TransformerLayer.Output(data=data, **aux_outputs)


class RepeatedTransformerLayer(Repeat):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.layer = TransformerLayer.default_config()
        return cfg

    def forward(
        self,
        data: Tensor,
        *,
        self_attention_mask: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> TransformerLayer.Output:
        def layer_fn(carry, x_i):
            layer_outputs: TransformerLayer.Output = self.layer(
                carry,
                self_attention_mask=self_attention_mask,
                cross_attention_data=cross_attention_data,
                cross_attention_mask=cross_attention_mask,
            )
            return layer_outputs.data, {
                k: v for k, v in layer_outputs._asdict().items() if k != "data"
            }

        repeat_outputs: Repeat.Output = self._run(layer_fn, data)
        ys = repeat_outputs.ys
        return TransformerLayer.Output(data=repeat_outputs.carry, **ys)


class PipelinedTransformerLayer(Pipeline):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("microbatch_size", -1, "The microbatch size.")
        cfg.layer = TransformerLayer.default_config()
        return cfg

    def forward(
        self,
        data: Tensor,
        *,
        self_attention_mask: Optional[Tensor] = None,
        cross_attention_data: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> TransformerLayer.Output:
        cfg = self.config

        carry_in = dict(data=data)
        # Even though masks do not change across layers, we include them in the carry so that they are aligned with the
        # microbatches.
        if self_attention_mask is not None:
            carry_in["self_attention_mask"] = self_attention_mask
        if cross_attention_data is not None:
            carry_in["cross_attention_data"] = cross_attention_data
        if cross_attention_mask is not None:
            carry_in["cross_attention_mask"] = cross_attention_mask

        carry_in = self._to_microbatches(carry_in, microbatch_size=cfg.microbatch_size)
        self.vlog(3, "carry_in=%s", shapes(carry_in))

        def layer_fn(carry, x_i):
            layer_outputs: TransformerLayer.Output = self.layer(**carry)
            carry.pop("data")
            return dict(**carry, data=layer_outputs.data), {
                k: v for k, v in layer_outputs._asdict().items() if k != "data"
            }

        pipeline_outputs: Pipeline.Output = self._run(layer_fn, carry_in)
        carry_out = self._from_microbatches(pipeline_outputs.carry["data"])

        ys = pipeline_outputs.ys
        self.vlog(3, "ys=%s", shapes(ys))
        # Take only the first microbatch for *_attention_probs.
        ys = jax.tree_map(lambda y: y[:, 0], ys)
        return TransformerLayer.Output(data=carry_out, **ys)

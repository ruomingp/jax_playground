import math

import copy
import numpy as np
import torch
from absl import logging
from absl.testing import absltest
import jax
from jax import nn
from jax import numpy as jnp

import attention
import module
from module import Module, InvocationContext
from attention import (
    MultiheadAttention,
    TransformerLayer,
    TransformerAttentionLayer,
    TransformerFeedForwardLayer,
)

# from torch.nn import MultiheadAttention, TransformerEncoderLayer, TransformerDecoderLayer
from transformers.models.roberta import modeling_roberta as hf_roberta


def _assert_allclose(a, b, atol=1e-6, rtol=1e-3):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=np.abs(a - b).max())


def _random_mask(prng_key, tgt_len, src_len):
    key1, key2 = jax.random.split(prng_key)
    mask = jnp.logical_not(
        jax.random.randint(key1, minval=0, maxval=2, shape=[tgt_len, src_len])
        +
        # Ensure that every tgt position attends to at least one src position, otherwise
        # torch_modules.MultiheadAttention will generate NaN.
        nn.one_hot(
            jax.random.randint(key2, minval=0, maxval=src_len, shape=[tgt_len]), src_len
        )
    )
    return mask.astype(jnp.float32) * -1e30


class MaskTest(absltest.TestCase):
    def testCausalMask(self):
        np.testing.assert_array_equal(
            [[False, True, True], [False, False, True], [False, False, False]],
            attention.make_causal_mask(3),
        )

    def testSegmentMask(self):
        np.testing.assert_array_equal(
            [
                [
                    [True, True, True, False],
                    [True, True, True, False],
                    [False, False, True, True],
                    [True, True, False, True],
                ]
            ],
            attention.make_segment_mask(
                target_segments=jnp.asarray([[1, 1, 2, 0]]),
                source_segments=jnp.asarray([[2, 2, 0, 1]]),
            ),
        )


class RelativePositionTest(absltest.TestCase):
    def testMTFBuckets(self):
        seq_len = 20
        # When number of buckets are limited, multiple relative positions share the same bucket.
        np.testing.assert_array_equal(
            [
                7,
                7,
                7,
                7,
                7,
                7,
                7,
                6,
                6,
                6,
                6,
                6,
                5,
                5,
                5,
                4,
                4,
                3,
                2,
                1,
                0,
                9,
                10,
                11,
                12,
                12,
                13,
                13,
                13,
                14,
                14,
                14,
                14,
                14,
                15,
                15,
                15,
                15,
                15,
                15,
                15,
            ],
            attention.t5_relative_position_bucket(
                jnp.arange(-seq_len, seq_len + 1, dtype=jnp.int32),
                num_buckets=16,
                max_distance=seq_len,
            ),
        )
        # When max_distance is limited, relative distances with magnitude >= max_distance share two buckets.
        np.testing.assert_array_equal(
            [
                13,
                13,
                13,
                13,
                13,
                13,
                13,
                12,
                11,
                11,
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                0,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                25,
                26,
                27,
                27,
                27,
                27,
                27,
                27,
                27,
            ],
            attention.t5_relative_position_bucket(
                jnp.arange(-seq_len, seq_len + 1, dtype=jnp.int32),
                num_buckets=28,
                max_distance=15,
            ),
        )


class MultiheadAttentionTest(absltest.TestCase):
    def testAllMask(self):
        return
        model_dim = 16
        num_heads = 4
        cfg = attention.MultiheadAttention.default_config().set(
            name="test",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
        )
        layer = cfg.instantiate(parent=None)
        batch_size, src_len, tgt_len = 2, 4, 6
        rng = np.random.default_rng(seed=123)
        query = jnp.asarray(rng.random([batch_size, tgt_len, model_dim]))
        key = jnp.asarray(rng.random([batch_size, src_len, model_dim]))
        value = jnp.asarray(rng.random([batch_size, src_len, model_dim]))
        mask = jnp.ones([batch_size, tgt_len, src_len], dtype=jnp.bool)
        layer_outputs = layer(query=query, key=key, value=value, mask=mask)
        layer_output_data = layer_outputs.data.detach()
        # No NaN.
        self.assertTrue(jnp.all(jnp.isfinite(layer_output_data)), layer_output_data)


def _attention_parameters(src: hf_roberta.RobertaSelfAttention):
    for name, param in src.named_parameters():
        print(f"{name}: {param.shape}")
    num_heads = src.self.num_attention_heads
    per_head_dim = src.self.attention_head_size
    results = {"attention": {}}
    for src_proj, dst_proj in (
        ("query", "q_proj"),
        ("key", "k_proj"),
        ("value", "v_proj"),
    ):
        dense = getattr(src.self, src_proj)
        # Note that torch.nn.Linear.weight is (output_dim, input_dim), so we need to transpose it before reshaping.
        dense_params = dict(
            weight=dense.weight.transpose(0, 1).view(-1, num_heads, per_head_dim),
            bias=dense.bias.view(num_heads, per_head_dim),
        )
        results["attention"][dst_proj] = dense_params
    output_dense = src.output.dense
    results["attention"]["o_proj"] = dict(
        weight=output_dense.weight.view(-1, num_heads, per_head_dim),
        bias=output_dense.bias,
    )
    norm = src.output.LayerNorm
    results["norm"] = dict(scale=norm.weight, bias=norm.bias)
    return jax.tree_map(lambda x: jnp.asarray(x.detach().numpy()), results)


def _copy_parameters(src: Module, dst: Module):
    with jnp.no_grad():
        for (src_name, src_param), (dst_name, dst_param) in zip(
            src.named_parameters(), dst.named_parameters()
        ):
            assert src_name == dst_name, f"{src_name} != {dst_name}"
            dst_param.copy_(src_param)


def _copy_transformer_parameters(src: Module, dst: attention.TransformerLayer):
    logging.debug(
        "src parameters: %s",
        [(name, param.shape) for name, param in src.named_parameters()],
    )
    _copy_parameters(src.norm1, dst.self_attention.norm)
    _copy_attention_parameters(src.self_attn, dst.self_attention.attention)
    if hasattr(src, "multihead_attn"):
        _copy_parameters(src.norm2, dst.cross_attention.norm)
        _copy_attention_parameters(src.multihead_attn, dst.cross_attention.attention)
        _copy_parameters(src.norm3, dst.feed_forward.norm)
    else:
        _copy_parameters(src.norm2, dst.feed_forward.norm)
    _copy_parameters(src.linear1, dst.feed_forward.linear1)
    _copy_parameters(src.linear2, dst.feed_forward.linear2)


class _BaseTest(absltest.TestCase):
    def _compare_attention_layers(
        self, ref: hf_roberta.RobertaAttention, layer: attention.MultiheadAttention
    ):
        layer_params = layer.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(0)
        )
        layer_param_shapes = jax.tree_map(lambda x: x.shape, layer_params)
        print(f"layer parameters={layer_param_shapes}")
        layer_params = _attention_parameters(ref)
        batch_size, src_len, tgt_len = 2, 4, 6
        model_dim, num_heads = layer.config.target_dim, layer.config.attention.num_heads
        rng = np.random.default_rng(seed=123)
        target = rng.random([batch_size, tgt_len, model_dim], dtype=np.float32)
        null_mask = jnp.zeros([tgt_len, src_len])
        rand_mask = _random_mask(jax.random.PRNGKey(123), tgt_len, src_len)
        # for mask in (None, null_mask, rand_mask):
        for mask in (None,):
            if mask is not None:
                mask = mask.unsqueeze(0).tile(batch_size, 1, 1).to(jnp.bool)
            layer_outputs: TransformerLayer.Output = layer(
                target=jnp.asarray(target),
                mask=mask,
                context=layer.make_invocation_context(
                    parameters=layer_params,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                ),
            )
            attn_mask = (
                None
                if mask is None
                else torch.as_tensor(mask).repeat_interleave(num_heads, dim=0)
            )
            (ref_outputs,) = ref.forward(
                torch.as_tensor(target, dtype=torch.float32),
                attention_mask=attn_mask,
                output_attentions=False,
            )
            _assert_allclose(layer_outputs.data, ref_outputs.detach().numpy())


class TransformerAttentionTest(_BaseTest):
    def testForward(self):
        model_dim = 16
        num_heads = 4
        cfg = attention.TransformerAttentionLayer.default_config().set(
            name="test",
            target_dim=model_dim,
            source_dim=model_dim,
            structure="postnorm",
        )
        cfg.attention.set(num_heads=num_heads)
        layer = cfg.instantiate(parent=None)
        roberta_config = hf_roberta.RobertaConfig(
            hidden_size=model_dim,
            num_attention_heads=num_heads,
            attention_probs_dropout_prob=0,
            hidden_dropout_prob=0,
            classifier_dropout=0,
        )
        print("roberta_config=%s" % roberta_config)
        ref = hf_roberta.RobertaAttention(roberta_config)
        self._compare_attention_layers(ref, layer)


if __name__ == "__main__":
    attention.enable_numeric_checks = True
    absltest.main()

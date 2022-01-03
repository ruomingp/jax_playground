import math

import copy
import numpy as np
from absl import logging
from absl.testing import absltest
import jax
from jax import nn
from jax import numpy as jnp

import layers
import module
from module import Module, InvocationContext
from attention import MultiheadAttention, TransformerLayer, TransformerAttentionLayer, TransformerFeedForwardLayer
from torch_modules import MultiheadAttention, TransformerEncoderLayer, TransformerDecoderLayer


def _assert_allclose(a, b, atol=1e-6, rtol=1e-3):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=np.abs(a - b).max())


def _random_mask(prng_key, tgt_len, src_len):
    key1, key2 = jax.random.split(prng_key)
    return jnp.logical_not(
        jax.random.randint(key1, minval=0, maxval=2, shape=[tgt_len, src_len]) +
        # Ensure that every tgt position attends to at least one src position, otherwise
        # torch_modules.MultiheadAttention will generate NaN.
        nn.one_hot(jax.random.randint(key2, minval=0, maxval=src_len, shape=[tgt_len]), src_len))


class MaskTest(absltest.TestCase):

    def testCausalMask(self):
        np.testing.assert_array_equal([[False, True, True], [False, False, True], [False, False, False]],
                                      layers.make_causal_mask(3))

    def testSegmentMask(self):
        np.testing.assert_array_equal(
            [[[True, True, True, False],
              [True, True, True, False],
              [False, False, True, True],
              [True, True, False, True]]],
            layers.make_segment_mask(target_segments=jnp.ndarray([[1, 1, 2, 0]]),
                                     source_segments=jnp.ndarray([[2, 2, 0, 1]])))


class RelativePositionTest(absltest.TestCase):

    def testMTFBuckets(self):
        seq_len = 20
        # When number of buckets are limited, multiple relative positions share the same bucket.
        np.testing.assert_array_equal(
            [7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 5, 4, 4,
             3, 2, 1, 0, 9, 10, 11, 12, 12, 13, 13, 13, 14, 14, 14, 14, 14,
             15, 15, 15, 15, 15, 15, 15],
            layers.t5_relative_position_bucket(
                jnp.arange(-seq_len, seq_len + 1, dtype=jnp.int64),
                num_buckets=16, max_distance=seq_len))
        # When max_distance is limited, relative distances with magnitude >= max_distance share two buckets.
        np.testing.assert_array_equal(
            [13, 13, 13, 13, 13, 13, 13, 12, 11, 11, 10, 9, 8, 7, 6, 5, 4,
             3, 2, 1, 0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 25, 26,
             27, 27, 27, 27, 27, 27, 27],
            layers.t5_relative_position_bucket(
                jnp.arange(-seq_len, seq_len + 1, dtype=jnp.int64),
                num_buckets=28, max_distance=15))


def _copy_attention_parameters(src: MultiheadAttention, dst: layers.MultiheadAttention):
    with jnp.no_grad():
        qw, kw, vw = src.in_proj_weight.transpose(0, 1).chunk(3, dim=-1)
        model_dim, num_heads, _ = dst.q_proj.weight.shape
        dst.q_proj.weight.copy_(qw.view(model_dim, num_heads, -1))
        dst.k_proj.weight.copy_(kw.view(model_dim, num_heads, -1))
        dst.v_proj.weight.copy_(vw.view(model_dim, num_heads, -1))
        qb, kb, vb = src.in_proj_bias.chunk(3)
        dst.q_proj.bias.copy_(qb.view(num_heads, -1))
        dst.k_proj.bias.copy_(kb.view(num_heads, -1))
        dst.v_proj.bias.copy_(vb.view(num_heads, -1))
        dst.o_proj.weight.copy_(src.out_proj.weight.view(model_dim, num_heads, -1))
        dst.o_proj.bias.copy_(src.out_proj.bias)


def _copy_parameters(src: Module, dst: Module):
    with jnp.no_grad():
        for (src_name, src_param), (dst_name, dst_param) in zip(src.named_parameters(), dst.named_parameters()):
            assert src_name == dst_name, f'{src_name} != {dst_name}'
            dst_param.copy_(src_param)


def _copy_transformer_parameters(src: Module, dst: layers.TransformerLayer):
    logging.debug('src parameters: %s', [(name, param.shape) for name, param in src.named_parameters()])
    with jnp.no_grad():
        _copy_parameters(src.norm1, dst.self_attention.norm)
        _copy_attention_parameters(src.self_attn, dst.self_attention.attention)
        if hasattr(src, 'multihead_attn'):
            _copy_parameters(src.norm2, dst.cross_attention.norm)
            _copy_attention_parameters(src.multihead_attn, dst.cross_attention.attention)
            _copy_parameters(src.norm3, dst.feed_forward.norm)
        else:
            _copy_parameters(src.norm2, dst.feed_forward.norm)
        _copy_parameters(src.linear1, dst.feed_forward.linear1)
        _copy_parameters(src.linear2, dst.feed_forward.linear2)


class _BaseTest(absltest.TestCase):

    def _compare_attention_layers(self, ref: MultiheadAttention, layer: layers.MultiheadAttention):
        _copy_attention_parameters(ref, layer)
        batch_size, src_len, tgt_len = 2, 4, 6
        model_dim, num_heads = layer.config.query_dim, layer.config.num_heads
        rng = np.random.default_rng(seed=123)
        query = jnp.ndarray(rng.random([batch_size, tgt_len, model_dim]))
        key = jnp.ndarray(rng.random([batch_size, src_len, model_dim]))
        value = jnp.ndarray(rng.random([batch_size, src_len, model_dim]))
        null_mask = jnp.zeros([tgt_len, src_len])
        rand_mask = _random_mask(tgt_len, src_len)
        for mask in (None, null_mask, rand_mask):
            if mask is not None:
                mask = mask.unsqueeze(0).tile(batch_size, 1, 1).to(jnp.bool)
            layer_outputs = layer(query=query, key=key, value=value, mask=mask)
            attn_mask = None if mask is None else mask.repeat_interleave(num_heads, dim=0)
            ref_outputs, ref_weights = ref(query, key, value, attn_mask=attn_mask)
            _assert_allclose(layer_outputs.data.detach(), ref_outputs.detach())
            _assert_allclose(layer_outputs.probs.mean(dim=1).detach(), ref_weights.detach())


class MultiheadAttentionTest(_BaseTest):

    def testForward(self):
        model_dim = 16
        num_heads = 4
        cfg = layers.MultiheadAttention.default_config().set(
            name='test', query_dim=model_dim, key_dim=model_dim, value_dim=model_dim, num_heads=num_heads)
        layer = cfg.instantiate(parent=None)
        ref = MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self._compare_attention_layers(ref, layer)

    def testAllMask(self):
        model_dim = 16
        num_heads = 4
        cfg = layers.MultiheadAttention.default_config().set(
            name='test', query_dim=model_dim, key_dim=model_dim, value_dim=model_dim, num_heads=num_heads)
        layer = cfg.instantiate(parent=None)
        batch_size, src_len, tgt_len = 2, 4, 6
        rng = np.random.default_rng(seed=123)
        query = jnp.ndarray(rng.random([batch_size, tgt_len, model_dim]))
        key = jnp.ndarray(rng.random([batch_size, src_len, model_dim]))
        value = jnp.ndarray(rng.random([batch_size, src_len, model_dim]))
        mask = jnp.ones([batch_size, tgt_len, src_len], dtype=jnp.bool)
        layer_outputs = layer(query=query, key=key, value=value, mask=mask)
        layer_output_data = layer_outputs.data.detach()
        # No NaN.
        self.assertTrue(jnp.all(jnp.isfinite(layer_output_data)), layer_output_data)


class TransformerLayerTest(_BaseTest):

    def _testSelfAttention(self, dropout: float = 0., dtype: jnp.dtype = jnp.float32, device: str = 'cpu:0'):
        device = jnp.device(device)
        model_dim = 16
        ff_hidden_dim = 64
        num_heads = 4
        ref = TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=ff_hidden_dim,
                                      dropout=dropout, batch_first=True, norm_first=True, dtype=dtype).to(device)
        cfg = layers.TransformerLayer.default_config().set(name='test', input_dim=model_dim, dtype=dtype)
        cfg.self_attention.attention.num_heads = num_heads
        cfg.self_attention.attention.dropout = dropout
        cfg.self_attention.dropout = dropout
        cfg.feed_forward.hidden_dim = ff_hidden_dim
        cfg.feed_forward.dropout = dropout
        layer = cfg.instantiate(parent=None)
        layer = layer.to(device)
        params = _copy_transformer_parameters(ref)

        batch_size, seq_len = 2, 4
        rng = np.random.default_rng(seed=123)
        orig_inputs = rng.random([batch_size, seq_len, model_dim])
        inputs = jnp.ndarray(orig_inputs).to(dtype).to(device)
        null_mask = jnp.zeros([seq_len, seq_len])
        causal_mask = layers.make_causal_mask(seq_len)
        rand_mask = _random_mask(seq_len, seq_len)
        for mask in (None, null_mask, causal_mask, rand_mask):
            if mask is None:
                src_mask = None
            else:
                mask = mask.to(jnp.bool).to(device=device).unsqueeze(0).tile((batch_size, 1, 1))
                src_mask = mask.repeat_interleave(num_heads, dim=0)
            context = InvocationContext(is_training=True, prng_key=jax.random.PRNGKey(123), parameters=params)
            with module.root_context(copy.deepcopy(context)):
                ref_outputs = ref(inputs, src_mask=src_mask).detach()
            idist.reinit_rng(123)
            ref_outputs2 = ref(inputs, src_mask=src_mask).detach()
            _assert_allclose(ref_outputs2, ref_outputs)
            with module.root_context(copy.deepcopy(context)):
                layer_outputs = layer(inputs, self_attention_mask=mask)
            layer_output_data = layer_outputs.data.detach()
            # forward() should not mutate 'inputs' in-place.
            _assert_allclose(inputs, orig_inputs)
            self.assertTrue(jnp.all(jnp.isfinite(layer_output_data)), layer_output_data)
            self.assertTrue(jnp.all(jnp.isfinite(ref_outputs)), ref_outputs)
            _assert_allclose(layer_output_data, ref_outputs)
            if mask is not rand_mask and not dropout:
                np.testing.assert_array_less(layer_outputs.self_attention_probs.data, 1. + 1e-6)
                _assert_allclose(jnp.sum(layer_outputs.self_attention_probs.data, dim=-1),
                                 jnp.ones([batch_size, num_heads, seq_len]))

    def testSelfAttention(self):
        self._testSelfAttention()

    def testSelfAttentionWithDropout(self):
        jnp.use_deterministic_algorithms(True)
        self._testSelfAttention(dropout=0.1)

    def testSelfAttentionFloat16(self):
        # TODO(r_pang): find a way to test the 16-bit float path properly.
        if jnp.cuda.is_available():
            self._testSelfAttention(dtype=jnp.float16, device='cuda:0')
        else:
            logging.info('testSelfAttentionFloat16 is skipped')

    def testCrossAttention(self, dtype=jnp.float32, device='cpu:0'):
        device = jnp.device(device)
        model_dim = 16
        ff_hidden_dim = 64
        num_heads = 4
        ref = TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=ff_hidden_dim,
                                      dropout=0., batch_first=True, norm_first=True, dtype=dtype, device=device)
        cfg = layers.TransformerLayer.default_config().set(name='test', input_dim=model_dim, dtype=dtype)
        cfg.self_attention.attention.num_heads = num_heads
        cfg.cross_attention = layers.TransformerAttentionLayer.default_config().set(source_dim=model_dim)
        cfg.cross_attention.attention.num_heads = num_heads
        cfg.feed_forward.hidden_dim = ff_hidden_dim
        layer = cfg.instantiate(parent=None)
        layer = layer.to(device)
        self._compare_attention_layers(ref.self_attn, layer.self_attention.attention)
        self._compare_attention_layers(ref.multihead_attn, layer.cross_attention.attention)
        _copy_transformer_parameters(ref, layer)

        batch_size, tgt_len, src_len = 2, 4, 6
        rng = np.random.default_rng(seed=123)
        inputs = jnp.ndarray(rng.random([batch_size, tgt_len, model_dim])).to(dtype).to(device)
        src_inputs = jnp.ndarray(rng.random([batch_size, src_len, model_dim])).to(dtype).to(device)
        causal_mask = layers.make_causal_mask(tgt_len).unsqueeze(0).tile((batch_size, 1, 1)).to(device=device)
        ref_outputs = ref(tgt=inputs, memory=src_inputs, tgt_mask=causal_mask.repeat_interleave(num_heads, dim=0))
        layer_outputs = layer(inputs, self_attention_mask=causal_mask, cross_attention_data=src_inputs)
        for probs in (layer_outputs.self_attention_probs, layer_outputs.cross_attention_probs):
            probs = probs.data
            np.testing.assert_array_less(probs, 1. + 1e-6)
            _assert_allclose(jnp.sum(probs, dim=-1), jnp.ones([batch_size, num_heads, tgt_len]))
        layer_output_data = layer_outputs.data.detach()
        ref_output_data = ref_outputs.detach()
        self.assertTrue(jnp.all(jnp.isfinite(layer_output_data)), layer_output_data)
        self.assertTrue(jnp.all(jnp.isfinite(ref_output_data)), ref_output_data)
        _assert_allclose(layer_output_data, ref_output_data)


if __name__ == "__main__":
    layers.enable_numeric_checks = True
    absltest.main()

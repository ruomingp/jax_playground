import jax
import numpy as np
import torch
from absl import logging
from absl.testing import absltest
from jax import nn
from jax import numpy as jnp
from transformers.models.roberta import modeling_roberta as hf_roberta

import attention
import layers
import module
from attention import (
    TransformerLayer,
    TransformerAttentionLayer,
)
from module import Module


def _assert_allclose(a, b, atol=1e-6, rtol=1e-3):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=np.abs(a - b).max())


def _shapes(nested_tensor):
    return jax.tree_map(lambda x: x.shape, nested_tensor)


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
        model_dim = 16
        num_heads = 4
        per_head_dim = model_dim // num_heads
        cfg = attention.MultiheadAttention.default_config().set(
            name="test",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
        )
        layer = cfg.instantiate(parent=None)

        layer_params = layer.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(123)
        )
        qkv_shapes = dict(
            weight=(model_dim, num_heads, per_head_dim), bias=(num_heads, per_head_dim)
        )
        self.assertEqual(
            {
                **{f"{x}_proj": qkv_shapes for x in ("q", "k", "v")},
                **{
                    "o_proj": dict(
                        weight=(model_dim, num_heads, per_head_dim), bias=(model_dim,)
                    ),
                    "dropout": {},
                },
            },
            _shapes(layer_params),
        )

        batch_size, src_len, tgt_len = 2, 4, 6
        rng = np.random.default_rng(seed=123)
        query = jnp.asarray(rng.random([batch_size, tgt_len, model_dim]))
        key = jnp.asarray(rng.random([batch_size, src_len, model_dim]))
        value = jnp.asarray(rng.random([batch_size, src_len, model_dim]))
        mask = jnp.ones([batch_size, tgt_len, src_len], dtype=jnp.bool_)
        context = layer.make_invocation_context(
            state=layer_params, is_training=True, prng_key=jax.random.PRNGKey(456)
        )
        layer_outputs = layer(
            query=query, key=key, value=value, mask=mask, context=context
        )
        layer_output_data = layer_outputs.data
        # No NaN.
        self.assertTrue(jnp.all(jnp.isfinite(layer_output_data)), layer_output_data)


def _to_jtensor(x):
    if isinstance(x, jnp.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return jnp.asarray(x.detach().numpy())
    return jax.tree_map(_to_jtensor, x)


def _parameters_from_layer_norm(src: torch.nn.LayerNorm):
    return _to_jtensor(dict(scale=src.weight, bias=src.bias))


def _parameters_from_roberta_attention(src: hf_roberta.RobertaAttention):
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
    results["norm"] = _parameters_from_layer_norm(src.output.LayerNorm)
    return _to_jtensor(results)


def _parameters_from_dense(dense: torch.nn.Linear):
    return _to_jtensor(dict(weight=dense.weight.transpose(0, 1), bias=dense.bias))


def _parameters_from_roberta_feed_forward(
    intermediate: hf_roberta.RobertaIntermediate, output: hf_roberta.RobertaOutput
):
    return _to_jtensor(
        dict(
            linear1=_parameters_from_dense(intermediate.dense),
            linear2=_parameters_from_dense(output.dense),
            norm=_parameters_from_layer_norm(output.LayerNorm),
        )
    )


def _parameters_from_roberta_layer(src: hf_roberta.RobertaLayer):
    return _to_jtensor(
        dict(
            self_attention=_parameters_from_roberta_attention(src.attention),
            feed_forward=_parameters_from_roberta_feed_forward(
                src.intermediate, src.output
            ),
        )
    )


def _as_torch_tensor(src: jnp.ndarray):
    return torch.as_tensor(np.asarray(src).copy())


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


class TransformerTest(absltest.TestCase):
    def _compare_against_roberta_attention(
        self, ref: hf_roberta.RobertaAttention, layer: TransformerAttentionLayer
    ):
        layer_params = layer.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(0)
        )
        layer_param_shapes = jax.tree_map(lambda x: x.shape, layer_params)
        print(f"layer state={layer_param_shapes}")
        layer_params = _parameters_from_roberta_attention(ref)
        batch_size, tgt_len = 2, 6
        model_dim, num_heads = layer.config.target_dim, layer.config.attention.num_heads
        rng = np.random.default_rng(seed=123)
        target = rng.random([batch_size, tgt_len, model_dim], dtype=np.float32)
        null_mask = jnp.zeros([tgt_len, tgt_len])
        rand_mask = _random_mask(jax.random.PRNGKey(123), tgt_len, tgt_len)
        for mask in (None, null_mask, rand_mask):
            if mask is not None:
                mask = mask[None, None, :, :].tile((batch_size, num_heads, 1, 1))
            layer_outputs: TransformerAttentionLayer.Output = layer(
                target=jnp.asarray(target),
                mask=mask,
                context=layer.make_invocation_context(
                    state=layer_params,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                ),
            )
            attn_mask = None if mask is None else _as_torch_tensor(mask)
            (ref_outputs,) = ref.forward(
                torch.as_tensor(target, dtype=torch.float32),
                attention_mask=attn_mask,
                output_attentions=False,
            )
            _assert_allclose(layer_outputs.data, ref_outputs.detach().numpy())

    def testAgainstRobertaAttention(self):
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
        self._compare_against_roberta_attention(ref, layer)

    def _compare_against_roberta_layer(
        self, ref: hf_roberta.RobertaLayer, layer: TransformerLayer
    ):
        layer_params = layer.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(0)
        )
        layer_param_shapes = jax.tree_map(lambda x: x.shape, layer_params)
        print(f"layer state={layer_param_shapes}")
        layer_params = _parameters_from_roberta_layer(ref)
        batch_size, tgt_len = 2, 6
        model_dim, num_heads = (
            layer.config.input_dim,
            layer.config.self_attention.attention.num_heads,
        )
        rng = np.random.default_rng(seed=123)
        target = rng.random([batch_size, tgt_len, model_dim], dtype=np.float32)
        null_mask = jnp.zeros([tgt_len, tgt_len])
        rand_mask = _random_mask(jax.random.PRNGKey(123), tgt_len, tgt_len)
        for mask in (None, null_mask, rand_mask):
            if mask is not None:
                mask = mask[None, None, :, :].tile((batch_size, num_heads, 1, 1))
            context = layer.make_invocation_context(
                state=layer_params,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
            )
            layer_outputs: TransformerLayer.Output = layer(
                jnp.asarray(target), self_attention_mask=mask, context=context
            )
            logging.info(
                "Auxiliary outputs=%s",
                context.output_collection.get_values_recursively(),
            )
            logging.info(
                "Summary=%s",
                _shapes(
                    context.output_collection.get_values_recursively(
                        module.OutputCollection.SECTION_SUMMARY
                    )
                ),
            )
            attn_mask = None if mask is None else _as_torch_tensor(mask)
            (ref_outputs,) = ref.forward(
                torch.as_tensor(target, dtype=torch.float32),
                attention_mask=attn_mask,
                output_attentions=False,
            )
            _assert_allclose(layer_outputs.data, ref_outputs.detach().numpy())

    def testAgainstRobertaLayer(self):
        model_dim = 16
        num_heads = 4
        cfg = TransformerLayer.default_config().set(name="test", input_dim=model_dim)
        cfg.self_attention.set(structure="postnorm")
        cfg.feed_forward.set(structure="postnorm", activation="nn.silu")
        cfg.feed_forward.linear1.set(bias=True)
        cfg.feed_forward.linear2.set(bias=True)
        cfg.self_attention.attention.set(num_heads=num_heads)
        cfg.self_attention.attention.input_linear.set(bias=True)
        cfg.self_attention.attention.output_linear.set(bias=True)
        layer: TransformerLayer = cfg.instantiate(parent=None)
        roberta_config = hf_roberta.RobertaConfig(
            hidden_size=model_dim,
            num_attention_heads=num_heads,
            attention_probs_dropout_prob=0,
            hidden_dropout_prob=0,
            classifier_dropout=0,
            # Jax's gelu uses an approximation by default and is slightly different from torch.nn.gelu.
            hidden_act="silu",
        )
        print("roberta_config=%s" % roberta_config)
        ref = hf_roberta.RobertaLayer(roberta_config)
        self._compare_against_roberta_layer(ref, layer)


if __name__ == "__main__":
    layers.enable_numeric_checks = True
    absltest.main()

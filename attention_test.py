import math

import jax
import numpy as np
import optax
import torch
from absl import logging
from absl.testing import absltest, parameterized
from jax import nn
from jax import numpy as jnp
from transformers.models.roberta import modeling_roberta as hf_roberta

import attention
import config as config_lib
import utils
from attention import (
    MultiheadLinearInit,
    PipelinedTransformerLayer,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerAttentionLayer,
    TransformerLayer,
)
from module import BaseLayer, FactorizationSpec, Module, ParameterPartitionSpec, PartitionSpec
from module import functional as F
from optimizer_base import OptParam
from optimizers import adafactor_optimizer
from test_utils import (
    TestCase,
    as_torch_tensor,
    assert_allclose,
    parameters_from_torch_layer,
    shapes,
)
from utils import flatten_items


def _random_mask(prng_key, tgt_len, src_len):
    key1, key2 = jax.random.split(prng_key)
    mask = jnp.logical_not(
        jax.random.randint(key1, minval=0, maxval=2, shape=[tgt_len, src_len])
        +
        # Ensure that every tgt position attends to at least one src position, otherwise
        # torch_modules.MultiheadAttention will generate NaN.
        nn.one_hot(jax.random.randint(key2, minval=0, maxval=src_len, shape=[tgt_len]), src_len)
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


class MultiheadLinearInitTest(absltest.TestCase):
    def testComputeFanAxes(self):
        shape = (4, 8, 4)  # H, W, I, O
        init_types = ["input", "output"]
        fan_specs = [
            {
                "fan_in": 4,
                "fan_out": 8 * 4,
                "fan_avg": 2 + 16,
            },
            {
                "fan_in": 8 * 4,
                "fan_out": 4,
                "fan_avg": 2 + 16,
            },
        ]
        for init_type, fans in zip(init_types, fan_specs):
            for dist in ("uniform", "normal", "truncated_normal"):
                for scale in (1.0, 2.0):
                    for fan_type in ("fan_in", "fan_out", "fan_avg"):
                        init: MultiheadLinearInit = (
                            MultiheadLinearInit.default_config()
                            .set(fan=fan_type, scale=scale, distribution=dist, type=init_type)
                            .instantiate()
                        )
                        fan = fans[fan_type]
                        weight = init.initialize(
                            "weight",
                            prng_key=jax.random.PRNGKey(1),
                            shape=shape,
                            dtype=jnp.float32,
                        )
                        self.assertEqual(weight.dtype, jnp.float32)
                        expected_std = scale / math.sqrt(fan)
                        actual_std = np.std(weight)
                        self.assertBetween(actual_std, expected_std / 1.5, expected_std * 1.5)


class MultiheadAttentionTest(TestCase):
    def testAllMask(self):
        utils.enable_numeric_checks = True

        model_dim = 12
        num_heads = 4
        per_head_dim = model_dim // num_heads
        cfg = attention.MultiheadAttention.default_config().set(
            name="test",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
        )
        layer: attention.MultiheadAttention = cfg.instantiate(parent=None)

        self.assertEqual(
            dict(
                dropout={},
                **{
                    proj: {
                        "weight": ParameterPartitionSpec(
                            shape=(model_dim, num_heads, per_head_dim),
                            partition=PartitionSpec(None, "model", None),
                            factorization=FactorizationSpec(axes=("row", None, "col")),
                        ),
                        "bias": ParameterPartitionSpec(
                            shape=(num_heads, per_head_dim),
                            partition=PartitionSpec("model", None),
                            factorization=None,
                        ),
                    }
                    for proj in ("q_proj", "k_proj", "v_proj")
                },
                o_proj={
                    "bias": ParameterPartitionSpec(
                        shape=(model_dim,),
                        partition=PartitionSpec(
                            None,
                        ),
                        factorization=None,
                    ),
                    "weight": ParameterPartitionSpec(
                        shape=(model_dim, num_heads, per_head_dim),
                        partition=PartitionSpec(None, "model", None),
                        factorization=FactorizationSpec(axes=("row", None, "col")),
                    ),
                },
            ),
            layer.create_partition_specs_recursively(),
        )

        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        qkv_shapes = dict(
            weight=(model_dim, num_heads, per_head_dim), bias=(num_heads, per_head_dim)
        )
        self.assertEqual(
            {
                **{f"{x}_proj": qkv_shapes for x in ("q", "k", "v")},
                **{
                    "o_proj": dict(weight=(model_dim, num_heads, per_head_dim), bias=(model_dim,)),
                    "dropout": {},
                },
            },
            shapes(layer_params),
        )

        batch_size, src_len, tgt_len = 2, 4, 6
        rng = np.random.default_rng(seed=123)
        query = jnp.asarray(rng.random([batch_size, tgt_len, model_dim]))
        key = jnp.asarray(rng.random([batch_size, src_len, model_dim]))
        value = jnp.asarray(rng.random([batch_size, src_len, model_dim]))
        mask = jnp.ones([batch_size, tgt_len, src_len], dtype=jnp.bool_)
        inputs = dict(query=query, key=key, value=value, mask=mask)
        layer_outputs, _ = F(
            layer,
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            inputs=inputs,
        )
        layer_output_data = layer_outputs.data
        # No NaN.
        self.assertTrue(jnp.all(jnp.isfinite(layer_output_data)), layer_output_data)
        utils.enable_numeric_checks = False

    @parameterized.parameters(jnp.float32, jnp.float16, jnp.bfloat16)
    def testDataTypes(self, dtype):
        model_dim = 16
        num_heads = 4
        cfg = attention.MultiheadAttention.default_config().set(
            name="test",
            query_dim=model_dim,
            key_dim=model_dim,
            value_dim=model_dim,
            num_heads=num_heads,
            dtype=dtype,
        )
        layer = cfg.instantiate(parent=None)

        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        batch_size, src_len, tgt_len = 2, 4, 6
        query = jnp.zeros([batch_size, tgt_len, model_dim], dtype=dtype)
        key = jnp.zeros([batch_size, src_len, model_dim], dtype=dtype)
        value = jnp.zeros([batch_size, src_len, model_dim], dtype=dtype)
        mask = jnp.ones([batch_size, tgt_len, src_len], dtype=jnp.bool_)
        inputs = dict(query=query, key=key, value=value, mask=mask)
        layer_outputs, _ = F(
            layer,
            state=layer_params,
            is_training=True,
            prng_key=jax.random.PRNGKey(456),
            inputs=inputs,
        )
        self.assertEqual(layer_outputs.data.dtype, dtype)


class TransformerTest(absltest.TestCase):
    def _compare_against_roberta_attention(
        self, ref: hf_roberta.RobertaAttention, layer: TransformerAttentionLayer
    ):
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        layer_param_shapes = jax.tree_map(lambda x: x.shape, layer_params)
        print(f"layer state={layer_param_shapes}")
        layer_params = parameters_from_torch_layer(ref)
        batch_size, tgt_len = 2, 6
        model_dim, num_heads = layer.config.target_dim, layer.config.attention.num_heads
        rng = np.random.default_rng(seed=123)
        target = rng.random([batch_size, tgt_len, model_dim], dtype=np.float32)
        null_mask = jnp.zeros([tgt_len, tgt_len])
        rand_mask = _random_mask(jax.random.PRNGKey(123), tgt_len, tgt_len)
        for mask in (None, null_mask, rand_mask):
            if mask is not None:
                mask = mask[None, None, :, :].tile((batch_size, num_heads, 1, 1))
            layer_outputs, _ = F(
                layer,
                inputs=dict(target=jnp.asarray(target), mask=mask),
                state=layer_params,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
            )
            attn_mask = None if mask is None else as_torch_tensor(mask)
            (ref_outputs,) = ref.forward(
                torch.as_tensor(target, dtype=torch.float32),
                attention_mask=attn_mask,
                output_attentions=False,
            )
            assert_allclose(layer_outputs.data, ref_outputs.detach().numpy())

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

    def _compare_against_roberta_layer(self, ref: hf_roberta.RobertaLayer, layer: TransformerLayer):
        layer_params = layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        layer_param_shapes = jax.tree_map(lambda x: x.shape, layer_params)
        print(f"layer state={layer_param_shapes}")
        layer_params = parameters_from_torch_layer(ref)
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
            layer_outputs, layer_output_collection = F(
                layer,
                inputs=dict(data=jnp.asarray(target), self_attention_mask=mask),
                state=layer_params,
                is_training=True,
                prng_key=jax.random.PRNGKey(0),
            )
            self.assertEqual(
                (num_heads, tgt_len, tgt_len), layer_outputs.self_attention_probs.shape
            )
            attn_mask = None if mask is None else as_torch_tensor(mask)
            (ref_outputs,) = ref.forward(
                torch.as_tensor(target, dtype=torch.float32),
                attention_mask=attn_mask,
                output_attentions=False,
            )
            assert_allclose(layer_outputs.data, ref_outputs.detach().numpy())

    def testAgainstRobertaLayer(self):
        model_dim = 16
        num_heads = 4
        cfg = TransformerLayer.default_config().set(name="test", input_dim=model_dim)
        cfg.self_attention.set(structure="postnorm")
        cfg.feed_forward.set(structure="postnorm", activation="nn.silu", hidden_dim=model_dim * 4)
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


class TestStackModel(BaseLayer):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("stack", None, "The transformer stack.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("stack", cfg.stack)

    def forward(self, data, self_attention_mask):
        # [batch, length, dim].
        x = self.stack(data, self_attention_mask=self_attention_mask).data
        x_mean = jnp.mean(x, axis=1, keepdims=True)
        # [batch, length].
        x_var = jnp.sum((x - x_mean) ** 2, axis=-1)
        loss = jnp.mean(x_var)
        return loss, {"mean": x_mean}


class StackedTransformerTest(TestCase):
    def _stack_config(self, stack_cls, *, num_layers, model_dim, num_heads, dtype):
        cfg = TestStackModel.default_config().set(
            name="test",
            stack=stack_cls.default_config().set(num_layers=num_layers, vlog=5, dtype=dtype),
        )
        layer_cfg = cfg.stack.layer
        layer_cfg.input_dim = model_dim
        layer_cfg.self_attention.attention.set(num_heads=num_heads)
        layer_cfg.feed_forward.hidden_dim = model_dim * 4
        layer_cfg.vlog = 5
        return cfg

    def testStackVsRepeat(self):
        self._compare_layers(StackedTransformerLayer, RepeatedTransformerLayer)

    def testStackVsRepeatBFloat16(self):
        # FIXME(rpang): fix the following test, which is caused by different behaviors of bfloat16 to float32 casting.
        # self._compare_layers(StackedTransformerLayer, RepeatedTransformerLayer, dtype=jnp.bfloat16)
        pass

    def testStackVsPipeline(self):
        self._compare_layers(StackedTransformerLayer, PipelinedTransformerLayer)

    def testRepeatVsPipeline(self):
        self._compare_layers(RepeatedTransformerLayer, PipelinedTransformerLayer)

    def _compare_layers(self, *layer_classes, dtype=jnp.float32):
        with utils.numeric_checks(False):
            batch_size, tgt_len = 10, 6
            num_layers, model_dim, num_heads = 3, 16, 4

            target = jax.random.normal(
                jax.random.PRNGKey(123), [batch_size, tgt_len, model_dim], dtype=dtype
            )
            rand_mask = _random_mask(jax.random.PRNGKey(123), tgt_len, tgt_len)
            rand_mask = rand_mask[None, None, :, :].tile((batch_size, num_heads, 1, 1))
            rand_mask = None

            all_params = []
            all_outputs = []
            all_summaries = []
            all_gradients = []
            all_updates = []
            for cls in layer_classes:
                cfg = self._stack_config(
                    cls,
                    num_layers=num_layers,
                    model_dim=model_dim,
                    num_heads=num_heads,
                    dtype=dtype,
                )
                if cls == PipelinedTransformerLayer:
                    cfg.stack.microbatch_size = 2
                layer: TestStackModel = cfg.instantiate(parent=None)

                param_partition_specs = layer.create_partition_specs_recursively()
                logging.info(
                    "%s.factorization_specs=%s",
                    cls,
                    jax.tree_map(lambda x: x.factorization, param_partition_specs),
                )
                layer_params = layer.initialize_parameters_recursively(
                    prng_key=jax.random.PRNGKey(123)
                )
                logging.info(
                    "%s.params=%s",
                    cls,
                    jax.tree_map(lambda x: f"{x.dtype}({x.shape})", layer_params),
                )

                def _loss(layer_params, data, mask):
                    layer_outputs, layer_output_collection = F(
                        layer,
                        inputs=dict(data=data, self_attention_mask=mask),
                        state=layer_params,
                        is_training=True,
                        prng_key=jax.random.PRNGKey(0),
                    )
                    loss, aux = layer_outputs
                    return loss, (aux, layer_output_collection)

                value, grads = jax.value_and_grad(_loss, has_aux=True)(
                    layer_params, jnp.asarray(target), rand_mask
                )
                loss, (aux, layer_output_collection) = value
                layer_outputs = (loss, aux)

                summaries = layer_output_collection.summaries
                logging.info(
                    "layer_outputs=%s summaries=%s",
                    shapes(flatten_items(layer_outputs)),
                    shapes(flatten_items(summaries)),
                )
                logging.info(
                    "global_grad_norm=%s, grads=%s",
                    optax.global_norm(grads),
                    shapes(flatten_items(grads)),
                )

                optimizer = adafactor_optimizer(learning_rate=0.1, clipping_threshold=1.0, eps=1e-2)
                opt_params = jax.tree_map(
                    lambda spec, p: OptParam(value=p, factorization_spec=spec.factorization),
                    param_partition_specs,
                    layer_params,
                )
                opt_state = optimizer.init(opt_params)
                logging.info("opt_state=%s", shapes(opt_state))
                updates, opt_state = optimizer.update(grads, opt_state, opt_params)

                def rms_norm(x):
                    return jnp.sqrt(jnp.mean(x**2))

                if cls == StackedTransformerLayer:
                    update_norms = jax.tree_map(rms_norm, updates)
                else:
                    update_norms = jax.vmap(lambda x: jax.tree_map(rms_norm, x))(updates)
                logging.info(
                    "global_update_norm=%s update_norms=%s",
                    optax.global_norm(updates),
                    dict(utils.flatten_items(update_norms)),
                )

                def recursive_stack(stacked):
                    return {
                        "layer": jax.tree_map(
                            lambda *xs: jnp.stack(xs, axis=0),
                            *[stacked[f"layer{i}"] for i in range(num_layers)],
                        )
                    }

                if cls == StackedTransformerLayer:
                    for x in (layer_params, grads, summaries, updates):
                        x["stack"] = recursive_stack(x["stack"])

                all_params.append(layer_params)
                all_outputs.append(layer_outputs)
                all_summaries.append(summaries)
                all_gradients.append(grads)
                all_updates.append(updates)

            self.assertNestedAllClose(all_params[0], all_params[1])
            self.assertNestedAllClose(all_summaries[0], all_summaries[1])
            self.assertNestedAllClose(all_outputs[0], all_outputs[1])
            self.assertNestedAllClose(all_gradients[0], all_gradients[1])
            self.assertNestedAllClose(all_updates[0], all_updates[1])


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()

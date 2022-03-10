import copy
import tempfile
from typing import Any, Dict, Union

import jax
import jax.random
import numpy as np
import torch
from absl import logging
from absl.testing import parameterized
from jax import numpy as jnp
from transformers.models.gpt2 import modeling_gpt2 as hf_gpt2
from transformers.models.roberta import modeling_roberta as hf_roberta
from transformers.models.vit import modeling_vit as hf_vit

from module import BaseLayer
from module import functional as F
from trainer import SpmdTrainer
from utils import flatten_items, shapes


def assert_allclose(a, b, atol=1e-6, rtol=1e-3, err_msg=""):
    a, b = jnp.asarray(a).astype(np.float32), jnp.asarray(b).astype(np.float32)
    np.testing.assert_allclose(
        a,
        b,
        atol=atol,
        rtol=rtol,
        err_msg=f"{err_msg}: {np.abs(a - b).max()}",
    )


def as_jax_tensor(x: Union[jnp.ndarray, torch.Tensor, Dict[str, Any]]):
    if isinstance(x, jnp.ndarray):
        return x
    if isinstance(x, np.ndarray):
        return jnp.asarray(x)
    if isinstance(x, torch.Tensor):
        t = x  # type: torch.Tensor
        return jnp.asarray(t.detach().numpy())
    return jax.tree_map(as_jax_tensor, x)


def as_torch_tensor(x: jnp.ndarray):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x.copy())
    if isinstance(x, jnp.ndarray):
        return torch.as_tensor(np.asarray(x).copy())
    return jax.tree_map(as_torch_tensor, x)


def parameters_from_torch_layer(src: Any):
    if isinstance(src, torch.nn.LayerNorm):
        dst = dict(scale=src.weight, bias=src.bias)
    elif isinstance(src, torch.nn.Linear):
        # torch.nn.Linear.weight uses layout (output, input) while ajax uses (input, output).
        dst = dict(weight=src.weight.transpose(0, 1), bias=src.bias)
    elif isinstance(src, torch.nn.Conv2d):
        # torch.nn.Conv2d.weight uses layout (output, input, H, W) while ajax uses (H, W, input, output).
        dst = dict(weight=src.weight.permute(2, 3, 1, 0), bias=src.bias)
    elif isinstance(src, hf_roberta.RobertaAttention):
        dst = _parameters_from_roberta_attention(src)
    elif isinstance(src, hf_roberta.RobertaLayer):
        dst = _parameters_from_roberta_layer(src)
    elif isinstance(src, hf_vit.ViTForImageClassification):
        dst = _parameters_from_vit_classification(src)
    elif isinstance(src, hf_vit.ViTLayer):
        dst = _parameters_from_vit_layer(src)
    elif isinstance(src, hf_gpt2.GPT2LMHeadModel):
        dst = _parameters_from_gpt2_layer(src)
    else:
        raise NotImplementedError(f"{type(src)}")
    return as_jax_tensor(dst)


def _parameters_from_attention_dense(
    src: Union[hf_roberta.RobertaSelfAttention, hf_vit.ViTSelfAttention],
    output: Union[hf_roberta.RobertaOutput, hf_vit.ViTOutput],
):
    num_heads = src.num_attention_heads
    per_head_dim = src.attention_head_size
    results = {}
    for src_proj, dst_proj in (
        ("query", "q_proj"),
        ("key", "k_proj"),
        ("value", "v_proj"),
    ):
        dense = getattr(src, src_proj)
        dense_params = parameters_from_torch_layer(dense)
        dense_params = dict(
            weight=dense_params["weight"].reshape(-1, num_heads, per_head_dim),
            bias=dense_params["bias"].reshape(num_heads, per_head_dim),
        )
        results[dst_proj] = dense_params
    output_dense = output.dense
    results["o_proj"] = dict(
        weight=output_dense.weight.view(-1, num_heads, per_head_dim),
        bias=output_dense.bias,
    )
    return results


def _parameters_from_roberta_attention(src: hf_roberta.RobertaAttention):
    return dict(
        attention=_parameters_from_attention_dense(src.self, src.output),
        norm=parameters_from_torch_layer(src.output.LayerNorm),
    )


def _parameters_from_roberta_feed_forward(
    intermediate: hf_roberta.RobertaIntermediate, output: hf_roberta.RobertaOutput
):
    return dict(
        linear1=parameters_from_torch_layer(intermediate.dense),
        linear2=parameters_from_torch_layer(output.dense),
        norm=parameters_from_torch_layer(output.LayerNorm),
    )


def _parameters_from_roberta_layer(src: hf_roberta.RobertaLayer):
    return dict(
        self_attention=_parameters_from_roberta_attention(src.attention),
        feed_forward=_parameters_from_roberta_feed_forward(
            src.intermediate,
            src.output,
        ),
    )


def _parameters_from_vit_classification(src: hf_vit.ViTForImageClassification):
    return dict(
        convert_to_sequence=_parameters_from_vit_embedding(src.vit.embeddings),
        encoder_1d=_parameters_from_vit_encoder(
            src.vit.encoder, src.vit.embeddings, src.vit.layernorm
        ),
        classifier=parameters_from_torch_layer(src.classifier),
    )


def _parameters_from_vit_embedding(src: hf_vit.ViTEmbeddings):
    return dict(conv=parameters_from_torch_layer(src.patch_embeddings.projection))


def _parameters_from_vit_encoder(
    src_enc: hf_vit.ViTEncoder, src_emb: hf_vit.ViTEmbeddings, src_norm: torch.nn.LayerNorm
):
    dst = dict(
        cls_token=src_emb.cls_token,
        input_dropout=dict(),
        transformer=dict(),
        pos_emb=dict(weight=src_emb.position_embeddings),
        output_norm=parameters_from_torch_layer(src_norm),
    )
    for layer_i, layer in enumerate(src_enc.layer):
        dst["transformer"][f"layer{layer_i}"] = parameters_from_torch_layer(layer)
    return dst


def _parameters_from_vit_attention(src: hf_vit.ViTAttention, src_norm: torch.nn.LayerNorm):
    return dict(
        attention=_parameters_from_attention_dense(src.attention, src.output),
        dropout=dict(),
        norm=parameters_from_torch_layer(src_norm),
    )


def _parameters_from_vit_feed_forward(
    intermediate: hf_vit.ViTIntermediate, output: hf_vit.ViTOutput, norm: torch.nn.LayerNorm
):
    return dict(
        linear1=parameters_from_torch_layer(intermediate.dense),
        linear2=parameters_from_torch_layer(output.dense),
        dropout1=dict(),
        dropout2=dict(),
        norm=parameters_from_torch_layer(norm),
    )


def _parameters_from_vit_layer(src: hf_vit.ViTLayer):
    return dict(
        self_attention=_parameters_from_vit_attention(src.attention, src.layernorm_before),
        feed_forward=_parameters_from_vit_feed_forward(
            src.intermediate, src.output, src.layernorm_after
        ),
    )


def _parameters_from_gpt2_feed_forward(src: hf_gpt2.GPT2MLP, norm: torch.nn.LayerNorm):
    return dict(
        norm=parameters_from_torch_layer(norm),
        linear1=dict(weight=src.c_fc.weight, bias=src.c_fc.bias),
        dropout1=dict(),
        linear2=dict(weight=src.c_proj.weight, bias=src.c_proj.bias),
        dropout2=dict(),
    )


def _parameters_from_gpt2_attention(src: hf_gpt2.GPT2Attention, norm: torch.nn.LayerNorm):
    # GPT2 attention weights are concat into one array, break out head and q/k/v dims.
    num_heads = src.num_heads
    attention = dict()
    # Head projection.
    c_attn_w = src.c_attn.weight.split(src.c_attn.weight.shape[-1] // 3, dim=-1)
    c_attn_b = src.c_attn.bias.split(src.c_attn.bias.shape[-1] // 3, dim=-1)
    for (w, b, proj) in zip(c_attn_w, c_attn_b, ("q_proj", "k_proj", "v_proj")):
        attention[proj] = dict(
            weight=w.reshape(w.shape[0], num_heads, -1), bias=b.reshape(num_heads, -1)
        )
    # Output projection.
    c_proj_w = src.c_proj.weight
    attention["o_proj"] = dict(
        weight=c_proj_w.T.reshape(c_proj_w.shape[0], num_heads, -1), bias=src.c_proj.bias
    )
    attention["dropout"] = dict()
    return dict(norm=parameters_from_torch_layer(norm), attention=attention, dropout=dict())


def _parameters_from_gpt2_block_layer(src: hf_gpt2.GPT2Block):
    return dict(
        self_attention=_parameters_from_gpt2_attention(src.attn, src.ln_1),
        feed_forward=_parameters_from_gpt2_feed_forward(src.mlp, src.ln_2),
    )


def _parameters_from_gpt2_layer(src: hf_gpt2.GPT2LMHeadModel):
    ref_transformer = src.transformer
    transformer = {
        f"layer{i}": _parameters_from_gpt2_block_layer(l) for i, l in enumerate(ref_transformer.h)
    }
    return dict(
        embedding_dropout=dict(),
        emb=dict(weight=ref_transformer.wte.weight),
        pos_emb=dict(weight=ref_transformer.wpe.weight[None, ...]),
        transformer=transformer,
        output_norm=parameters_from_torch_layer(ref_transformer.ln_f),
        output_dropout=dict(),
    )


class TestCase(parameterized.TestCase):
    def _compute_layer_outputs(
        self, *, test_layer: BaseLayer, ref_layer: Any, test_inputs: Any, ref_inputs: Any
    ):
        layer_params = test_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(0))
        for name, param in flatten_items(layer_params):
            logging.info("Test: %s=%s", name, param.shape)

        for name, param in ref_layer.named_parameters():
            logging.info("Ref: %s=%s", name, param.shape)

        params_from_ref = parameters_from_torch_layer(ref_layer)
        self.assertCountEqual(
            flatten_items(shapes(params_from_ref)), flatten_items(shapes(layer_params))
        )
        del layer_params

        test_outputs, _ = F(
            test_layer,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=params_from_ref,
            inputs=test_inputs,
        )
        ref_layer.eval()
        ref_outputs = ref_layer(ref_inputs)
        return test_outputs, ref_outputs

    def assertNestedAllClose(self, a, b, atol=1e-6, rtol=1e-3):
        a_items = flatten_items(a)
        b_items = flatten_items(b)
        self.assertEqual([name for name, _ in a_items], [name for name, _ in b_items])
        for (a_name, a_value), (b_name, b_value) in zip(a_items, b_items):
            self.assertEqual(a_name, b_name)
            if isinstance(a_value, jnp.ndarray) or isinstance(b_value, jnp.ndarray):
                a_value, b_value = jnp.asarray(a_value), jnp.asarray(b_value)
                self.assertEqual(a_value.dtype, b_value.dtype, msg=f"{a_name}")
                self.assertEqual(a_value.shape, b_value.shape, msg=f"{a_name}")
                assert_allclose(a_value, b_value, atol=atol, rtol=rtol, err_msg=f"{a_name}")
            else:
                self.assertAlmostEqual(a_value, b_value)


class TrainerConfigTestCase(TestCase):
    """Base class for testing trainer configs."""

    def _test_with_trainer_config(self, trainer_config, mesh_size: Dict[str, int] = {}):
        cfg = copy.deepcopy(trainer_config)
        cfg.dir = cfg.dir or tempfile.mkdtemp()
        cfg.mesh_axis_names = cfg.mesh_axis_names or ("data", "model")
        cfg.mesh_shape = cfg.mesh_shape or (len(jax.devices()), 1)
        cfg.max_step = 3
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        trainer.run(jax.random.PRNGKey(123))

        state_spec_map = dict(flatten_items(trainer.trainer_state_specs))
        for path, value in flatten_items(trainer.trainer_state):
            state_spec = state_spec_map.get(path)
            logging.info(
                "State: %s=%s(%s) state_spec=%s", path, value.dtype, value.shape, state_spec
            )
            if state_spec is None:
                continue
            self.assertSequenceEqual(value.shape, state_spec.shape)
            self.assertLen(
                state_spec.partition, len(value.shape), msg=f"{path}: {state_spec} vs {value.shape}"
            )
            for dim_size, dim_name in zip(value.shape, state_spec.partition):
                if dim_name is None:
                    continue
                mesh_dim_size = mesh_size.get(dim_name, 1)
                self.assertEqual(
                    dim_size % mesh_dim_size,
                    0,
                    msg=f"{path}: {dim_size} % {mesh_dim_size} != 0 for {dim_name} in {value.shape} vs. {state_spec}",
                )

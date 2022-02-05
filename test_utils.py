from typing import Any, Dict, Union

import jax
import numpy as np
import torch
from absl import logging
from absl.testing import parameterized
from jax import numpy as jnp
from transformers.models.roberta import modeling_roberta as hf_roberta
from transformers.models.vit import modeling_vit as hf_vit

from module import BaseLayer
from module import functional as F
from utils import flatten_items, shapes


def assert_allclose(a, b, atol=1e-6, rtol=1e-3):
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=np.abs(a - b).max())


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
        pos_emb=dict(weight=src_emb.position_embeddings),
        output_norm=parameters_from_torch_layer(src_norm),
    )
    for layer_i, layer in enumerate(src_enc.layer):
        dst[f"layer{layer_i}"] = parameters_from_torch_layer(layer)
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


class TestCase(parameterized.TestCase):
    def _compute_layer_outputs(
        self, *, test_layer: BaseLayer, ref_layer: Any, test_inputs: Any, ref_inputs: Any
    ):
        param_specs = test_layer.create_parameter_specs_recursively()
        for name, param_spec in flatten_items(param_specs):
            logging.info("Test: %s=%s", name, param_spec.shape)
        layer_params = test_layer.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(0), param_specs=param_specs
        )

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

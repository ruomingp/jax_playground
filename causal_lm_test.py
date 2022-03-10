import jax
import numpy as np
from absl.testing import absltest
from transformers.models.gpt2 import modeling_gpt2 as hf_gpt2

import causal_lm
import utils
from attention import StackedTransformerLayer
from module import functional as F
from test_utils import TestCase, as_torch_tensor, assert_allclose


class Gpt2TransformerTest(TestCase):
    def testTiedLmHeadDiffersFromUntied(self):
        utils.enable_numeric_checks = True

        hidden_dim = 16
        vocab_size = 24
        source_length = 11

        transformer_cfg = causal_lm.gpt2_transformer_cfg(
            StackedTransformerLayer, num_layers=2, hidden_dim=hidden_dim, num_heads=4
        )
        shared_model_kwargs = dict(
            vocab_size=vocab_size,
            source_length=source_length,
            hidden_dim=hidden_dim,
            transformer=transformer_cfg,
        )
        tied_head = (
            causal_lm.Model.default_config()
            .set(name="test_tied", lm_head=None, **shared_model_kwargs)
            .instantiate(parent=None)
        )
        tied_head_state = tied_head.initialize_parameters_recursively(jax.random.PRNGKey(0))
        assert tied_head_state.get("lm_head") is None
        untied_head = (
            causal_lm.Model.default_config()
            .set(
                name="test_untied", lm_head=causal_lm.LmHead.default_config(), **shared_model_kwargs
            )
            .instantiate(parent=None)
        )
        untied_head_state = untied_head.initialize_parameters_recursively(jax.random.PRNGKey(0))
        assert untied_head_state.get("lm_head") is not None
        inputs = jax.random.randint(
            jax.random.PRNGKey(1), minval=1, maxval=vocab_size, shape=(3, source_length)
        )

        # Test values.
        def layer_output(state, layer):
            return F(
                layer,
                inputs=dict(inputs=inputs, return_aux=True),
                state=state,
                is_training=False,
                prng_key=jax.random.PRNGKey(2),
            )[0][1]["logits"]

        tied_logits = layer_output(tied_head_state, tied_head)
        untied_logits = layer_output(untied_head_state, untied_head)
        np.testing.assert_raises(AssertionError, assert_allclose, tied_logits, untied_logits)

        # Test grads.
        def layer_grad(state, layer):
            return layer_output(state, layer).sum()

        def check_grads(tied_state, untied_state):
            tied_head_grad = jax.grad(layer_grad)(tied_state, tied_head)["emb"]["weight"]
            untied_head_grad = jax.grad(layer_grad)(untied_state, untied_head)["emb"]["weight"]
            np.testing.assert_raises(
                AssertionError, assert_allclose, tied_head_grad, untied_head_grad
            )

        # Assert grad is different tied vs untied
        check_grads(tied_head_state, untied_head_state)
        # Set untied head weight to tied lm_head value and check again.
        untied_head_state["lm_head"]["weight"] = tied_head_state["emb"]["weight"].clone()
        check_grads(tied_head_state, untied_head_state)

        utils.enable_numeric_checks = False

    def testAgainstHfGpt2Lm(self):
        hidden_dim = 16
        vocab_size = 24
        num_heads = 4
        num_layers = 2
        source_length = 11
        # Reference implementation.
        ref_cfg = hf_gpt2.GPT2Config(
            n_embd=hidden_dim,
            n_head=num_heads,
            n_layer=num_layers,
            n_positions=source_length,
            vocab_size=vocab_size,
        )
        ref_layer = hf_gpt2.GPT2LMHeadModel(ref_cfg).eval()
        # Equivalent ajax implementation.
        transformer_cfg = causal_lm.gpt2_transformer_cfg(
            StackedTransformerLayer,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        layer = (
            causal_lm.Model.default_config()
            .set(
                dropout_rate=0.0,  # Disables dropout throughout.
                vocab_size=vocab_size,
                source_length=source_length,
                hidden_dim=hidden_dim,
                transformer=transformer_cfg,
                name="layer_test",
            )
            .instantiate(parent=None)
        )
        inputs = np.random.randint(1, vocab_size, size=(3, source_length))
        (_, test_aux), ref_outputs = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=ref_layer,
            test_inputs=dict(inputs=inputs, return_aux=True),
            ref_inputs=as_torch_tensor(inputs),
        )
        test_logits = test_aux["logits"]
        ref_logits = ref_outputs.logits.detach().numpy()
        assert_allclose(test_logits, ref_logits)


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()

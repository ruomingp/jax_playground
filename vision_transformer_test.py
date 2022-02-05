import numpy as np
from absl import logging
from absl.testing import absltest
from transformers.models.vit import configuration_vit as hf_vit_config
from transformers.models.vit import modeling_vit as hf_vit

from test_utils import TestCase, as_jax_tensor, as_torch_tensor, assert_allclose
from vision_transformer import Model


class ModelTest(TestCase):
    def testModel(self):
        model_dim, ff_dim, num_heads, num_layers = 12, 24, 2, 3

        # Create the test model.
        cfg = Model.default_config().set(name="test")
        cfg.convert_to_sequence.set(patch_size=(16, 16))
        cfg.encoder_1d.pos_emb.shape = [14 * 14 + 1]  # 14 = 224 / 16. +1 for cls_token.
        cfg.encoder_1d.set(
            input_dim=model_dim, num_layers=num_layers, global_feature_extraction="cls_token"
        )
        transformer_cfg = cfg.encoder_1d.transformer
        transformer_cfg.self_attention.attention.num_heads = num_heads
        transformer_cfg.feed_forward.hidden_dim = ff_dim
        # Disable dropout for deterministic behavior.
        cfg.encoder_1d.input_dropout.rate = 0.0
        transformer_cfg.feed_forward.dropout.rate = 0.0
        transformer_cfg.self_attention.dropout.rate = 0.0
        transformer_cfg.self_attention.attention.dropout.rate = 0.0
        logging.info("Test config:\n%s", cfg.debug_string())
        model: Model = cfg.instantiate(parent=None)

        # Create the ref model.
        ref_cfg = hf_vit_config.ViTConfig(
            hidden_size=model_dim,
            num_attention_heads=num_heads,
            intermediate_size=ff_dim,
            num_hidden_layers=num_layers,
            num_labels=1000,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
            layer_norm_eps=1e-6,
        )
        ref_model = hf_vit.ViTForImageClassification(ref_cfg)

        batch_size = 2
        inputs = {
            "image": np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32),
            "label": np.random.randint(0, 999, [batch_size]).astype(np.int32),
        }
        ref_inputs = as_torch_tensor(inputs["image"]).permute(0, 3, 1, 2)
        (test_loss, test_aux), ref_outputs = self._compute_layer_outputs(
            test_layer=model,
            ref_layer=ref_model,
            test_inputs=as_jax_tensor(inputs),
            ref_inputs=ref_inputs,
        )

        test_logits = test_aux["logits"]
        ref_logits = ref_outputs.logits.detach().numpy()
        assert_allclose(test_logits, ref_logits)


if __name__ == "__main__":
    absltest.main()

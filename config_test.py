import collections
import copy
import dataclasses

import numpy as np
import tensorflow_datasets as tfds
from absl.testing import absltest
from jax import numpy as jnp

import config


class ConfigTest(absltest.TestCase):
    def testDefine(self):
        cfg = config.Config()
        self.assertEqual(0, len(cfg))
        self.assertEmpty(cfg.keys())
        self.assertEmpty(cfg.items())
        self.assertNotIn("num_layers", cfg)
        cfg.define("num_layers", 10, "The number of layers.")
        self.assertEqual(1, len(cfg))
        self.assertEqual(["num_layers"], cfg.keys())
        self.assertEqual([("num_layers", 10)], cfg.items())
        self.assertIn("num_layers", cfg)
        self.assertEqual(10, cfg.num_layers)
        # Duplicate definition.
        with self.assertRaisesRegex(config.FieldAlreadyExistsError, ".*num_layers.*"):
            cfg.define("num_layers", 10, "The number of layers (defined again)")
        # Field name cannot contain '.'.
        with self.assertRaisesRegex(config.InvalidConfigNameError, "hidden.dim"):
            cfg.define("hidden.dim", 10, "The hidden dimension.")
        # Field value can be a function.
        cfg.define("func", lambda x: jnp.maximum(x, 0), "The activation function.")

    def testSet(self):
        cfg = config.Config()
        cfg.define("num_layers", 10, "The number of layers.")
        cfg.define("hidden_dim", 16, "The hidden dimension size.")
        # Set via setattr.
        cfg.num_layers = 6
        self.assertEqual(6, cfg.num_layers)
        self.assertEqual(16, cfg.hidden_dim)
        # Set() can update multiple fields.
        self.assertIs(cfg.set(num_layers=8, hidden_dim=128), cfg)
        self.assertEqual(8, cfg.num_layers)
        self.assertEqual(128, cfg.hidden_dim)
        self.assertEqual([("hidden_dim", 128), ("num_layers", 8)], cfg.items())
        self.assertEqual("\n".join(["hidden_dim: 128", "num_layers: 8"]), cfg.debug_string())
        # UnknownFieldError.
        with self.assertRaisesRegex(
            config.UnknownFieldError, r"keys are \['hidden_dim', 'num_layers'\].*"
        ):
            cfg.vocab_size = 1024
        # When the unknown field is close enough to a defined field.
        with self.assertRaisesRegex(config.UnknownFieldError, r".*did you mean: \[num_layers\].*"):
            cfg.num_layer = 5

    def testValueTypes(self):
        cfg = config.Config()
        cfg.define("sub", None, "The sub layers.")
        # Config field values can be a list.
        cfg.sub = [None, 123, "str", np.float64]
        self.assertEqual(
            "\n".join(
                [
                    "sub[0]: None",
                    "sub[1]: 123",
                    "sub[2]: 'str'",
                    "sub[3]: 'numpy.float64'",
                ]
            ),
            cfg.debug_string(),
        )
        # Config field values can be a tuple.
        cfg.sub = (None, 123, "str", np.float64)
        self.assertEqual(
            "\n".join(
                [
                    "sub[0]: None",
                    "sub[1]: 123",
                    "sub[2]: 'str'",
                    "sub[3]: 'numpy.float64'",
                ]
            ),
            cfg.debug_string(),
        )
        # Config field values can be a dict.
        cfg.sub = dict(none=None, int=123, str="str", type=np.float64)
        self.assertEqual(
            "\n".join(
                [
                    "sub['none']: None",
                    "sub['int']: 123",
                    "sub['str']: 'str'",
                    "sub['type']: 'numpy.float64'",
                ]
            ),
            cfg.debug_string(),
        )
        # Config field values can be a named tuple.
        ntuple = collections.namedtuple("ntuple", ("none", "int", "str", "type"))
        cfg.sub = ntuple(none=None, int=123, str="str", type=np.float64)
        self.assertEqual(
            "\n".join(
                [
                    "sub['none']: None",
                    "sub['int']: 123",
                    "sub['str']: 'str'",
                    "sub['type']: 'numpy.float64'",
                ]
            ),
            cfg.debug_string(),
        )

        # Config field values can be a dataclass.
        @dataclasses.dataclass
        class dclass:
            int_val: int
            str_val: "str"
            type_val: np.dtype

        cfg.sub = dclass(int_val=123, str_val="str", type_val=np.float64)
        self.assertEqual(
            "\n".join(
                [
                    "sub['int_val']: 123",
                    "sub['str_val']: 'str'",
                    "sub['type_val']: 'numpy.float64'",
                ]
            ),
            cfg.debug_string(),
        )

    def testNestedConfigs(self):
        model_cfg = config.Config()
        model_cfg.define("encoder", config.Config(), "The encoder.")
        model_cfg.define("decoder", config.Config(), "The decoder.")
        enc_cfg = model_cfg.encoder
        enc_cfg.define("num_layers", 12, "The number of layers.")
        dec_cfg = model_cfg.decoder
        dec_cfg.define("num_layers", 8, "The number of layers.")
        dec_cfg.define("vocab_size", 256, "The output vocab size.")
        self.assertEqual(8, model_cfg.decoder.num_layers)
        self.assertEqual(
            "\n".join(
                [
                    "decoder.num_layers: 8",
                    "decoder.vocab_size: 256",
                    "encoder.num_layers: 12",
                ]
            ),
            model_cfg.debug_string(),
        )

        cfg2 = copy.deepcopy(model_cfg)
        self.assertEqual(8, cfg2.decoder.num_layers)
        cfg2.decoder.num_layers = 16
        self.assertEqual(16, cfg2.decoder.num_layers)
        # The original model_cfg remain unchanged.
        self.assertEqual(8, model_cfg.decoder.num_layers)

        # Freeze cfg2.
        self.assertIs(cfg2.freeze(), cfg2)
        with self.assertRaisesRegex(config.FrozenConfigError, r".*encoder=None"):
            cfg2.encoder = None
        with self.assertRaisesRegex(config.FrozenConfigError, r".*num_layers=32"):
            cfg2.decoder.num_layers = 32

    def testInstantiableConfigForConfigurable(self):
        class Layer(config.Configurable):
            @classmethod
            def default_config(cls):
                cfg = config.InstantiableConfig(cls)
                cfg.define("input_dim", 8, "The input dim.")
                cfg.define("output_dim", 16, "The output dim.")
                return cfg

            def __init__(self, cfg):
                self.cfg = cfg

        cfg = Layer.default_config()
        layer1 = cfg.instantiate()
        cfg2 = copy.deepcopy(cfg)
        layer2 = cfg2.instantiate()
        self.assertEqual(layer1.cfg.debug_string(), layer2.cfg.debug_string())

    def testInstantiableConfigFromInitSignature(self):
        # Generate the config from the signature of Layer.__init__().
        class Layer:
            def __init__(self, in_features: int, out_features: int, bias: bool = True):
                self.params = {}
                self.params["weight"] = np.random.normal(size=(in_features, out_features))
                if bias:
                    self.params["bias"] = np.zeros(shape=(out_features,))

            def named_parameters(self):
                return self.params.items()

        cfg = config.config_for_class(Layer)
        self.assertContainsSubset({"cls", "in_features", "out_features", "bias"}, cfg.keys())
        self.assertEqual(cfg.cls, Layer)
        self.assertTrue(
            cfg.bias
        )  # the config default value is the same as the __init__ argument default value.
        cfg.in_features = 8
        cfg.out_features = 16
        layer1 = cfg.instantiate()
        cfg2 = copy.deepcopy(cfg)
        layer2 = cfg2.instantiate()

        def param_shapes(layer):
            return [(name, param.shape) for name, param in layer.named_parameters()]

        self.assertEqual(param_shapes(layer1), param_shapes(layer2))

    def testInstantiableConfigFromFunctionSignature(self):
        cfg = config.config_for_function(tfds.load)
        self.assertContainsSubset({"fn", "name", "split", "download"}, cfg.keys())

        def fn_with_args(*args):
            return list(args)

        cfg = config.config_for_function(fn_with_args)
        cfg.args = [1, 2, 3]
        self.assertEqual([1, 2, 3], cfg.instantiate())


if __name__ == "__main__":
    absltest.main()

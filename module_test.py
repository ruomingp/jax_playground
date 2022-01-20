from functools import partial

import jax.random
from absl.testing import absltest
from jax import numpy as jnp

import config as config_lib
import param_init
from module import (
    BaseLayer,
    InvocationContext,
    Module,
    NestedParameterSpec,
    OutputCollection,
    ParameterSpec,
)
from module import (
    new_output_collection,
    current_context,
    set_current_context,
    functional as F,
)


class OutputCollectionTest(absltest.TestCase):
    def testOutputCollectionChildren(self):
        c = new_output_collection()
        c.summaries["x"] = 1
        c1 = c.add_child("c1")
        c1.summaries["y"] = 2
        c2 = c.add_child("c2")
        c2.summaries["z"] = 3
        self.assertEqual({"x": 1, "c1": {"y": 2}, "c2": {"z": 3}}, c.summaries)


class InvocationContextTest(absltest.TestCase):
    def testContextOutputCollection(self):
        context = InvocationContext(
            module=None,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state={"x": 1},
            output_collection=new_output_collection(),
        )
        context.add_summary("x", 1)
        context.add_summary("y", 2)
        context.add_state_update("z", 3)
        self.assertEqual({"x": 1, "y": 2}, context.get_summaries())
        self.assertEqual({"z": 3}, context.get_state_updates())

    def testContextStack(self):
        context1 = InvocationContext(
            module=None,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state={"x": 1},
            output_collection=new_output_collection(),
        )
        with set_current_context(context1):
            self.assertIs(current_context(), context1)
            self.assertEqual(current_context().state["x"], 1)
            context2 = InvocationContext(
                module=None,
                is_training=True,
                prng_key=jax.random.PRNGKey(123),
                state={"x": 2},
                output_collection=new_output_collection(),
            )
            with set_current_context(context2):
                self.assertIs(current_context(), context2)
                self.assertEqual(current_context().state["x"], 2)

            # No longer in context2, but still in context1.
            self.assertIs(current_context(), context1)
            self.assertEqual(current_context().state["x"], 1)


class TestLayer(BaseLayer):
    def create_parameter_specs_recursively(self) -> NestedParameterSpec:
        return {
            "moving_mean": ParameterSpec(
                shape=[],
                partition_spec=None,
                dtype=jnp.float32,
                initializer=param_init.ConstantInitializer(value=1.0),
            )
        }

    def forward(self, x):
        self.add_summary("x", x)
        self.add_state_update("moving_mean", 0.1 * x + 0.9 * self.state["moving_mean"])
        return x - self.state["moving_mean"]


class TestParentLayer(BaseLayer):
    """A parent layer."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("child", TestLayer.default_config(), "The child layer config.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("child", cfg.child)

    def forward(self, x):
        return self.child(x)


class ModuleTest(absltest.TestCase):
    def testForward(self):
        test_module: TestLayer = (
            TestLayer.default_config().set(name="test").instantiate(parent=None)
        )
        state = test_module.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(456)
        )
        self.assertEqual({"moving_mean": 1.0}, state)
        y, output_collection = jax.jit(partial(F, test_module, is_training=True))(
            prng_key=jax.random.PRNGKey(123), state=state, inputs=(jnp.asarray(5.0),)
        )
        self.assertEqual(4, y)
        self.assertEqual(
            OutputCollection(summaries={"x": 5}, state_updates={"moving_mean": 1.4}),
            output_collection,
        )

    def testParentForward(self):
        test_module: TestParentLayer = (
            TestParentLayer.default_config().set(name="test").instantiate(parent=None)
        )
        state = test_module.initialize_parameters_recursively(
            prng_key=jax.random.PRNGKey(456)
        )
        self.assertEqual({"child": {"moving_mean": 1.0}}, state)
        y, output_collection = jax.jit(partial(F, test_module, is_training=True))(
            prng_key=jax.random.PRNGKey(123), state=state, inputs=(jnp.asarray(5.0),)
        )
        self.assertEqual(4, y)
        self.assertEqual(
            OutputCollection(
                summaries={"child": {"x": 5}},
                state_updates={"child": {"moving_mean": 1.4}},
            ),
            output_collection,
        )


if __name__ == "__main__":
    absltest.main()

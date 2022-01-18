import jax.random
from absl.testing import absltest
from module import OutputCollection, Module, InvocationContext, current_context, set_current_context, functional as F


class OutputCollectionTest(absltest.TestCase):
    def testOutputCollectionSections(self):
        c = OutputCollection()
        c.add_value('x', 1, section=OutputCollection.SECTION_SUMMARY)
        c.add_value('y', 2, section=OutputCollection.SECTION_SUMMARY)
        c.add_value('z', 3, section=OutputCollection.SECTION_STATE_UPDATE)
        self.assertEqual({"x": 1, "y": 2}, c.get_values_recursively(OutputCollection.SECTION_SUMMARY))
        self.assertEqual({"z": 3}, c.get_values_recursively(OutputCollection.SECTION_STATE_UPDATE))

    def testOutputCollectionChildren(self):
        c = OutputCollection()
        c.add_value('x', 1, section=OutputCollection.SECTION_SUMMARY)
        c1 = c.add_child('c1')
        c1.add_value('y', 2, section=OutputCollection.SECTION_SUMMARY)
        c2 = c.add_child('c2')
        c2.add_value('z', 3, section=OutputCollection.SECTION_SUMMARY)
        self.assertEqual({"x": 1, "c1": {"y": 2}, "c2": {"z": 3}},
                         c.get_values_recursively(OutputCollection.SECTION_SUMMARY))


class TestModule(Module):
    pass


class InvocationContextTest(absltest.TestCase):
    def testRootContext(self):
        test_module: TestModule = TestModule.default_config().set(name="test").instantiate(parent=None)
        context = InvocationContext(module=test_module, is_training=True, prng_key=jax.random.PRNGKey(123),
                                    state={'x': 1}, output_collection=OutputCollection())


if __name__ == "__main__":
    absltest.main()

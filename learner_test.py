import jax.nn
import numpy as np
import optax
from absl.testing import absltest, parameterized
from jax import numpy as jnp

import utils
from config import config_for_function
from learner import Learner, LearnerState, sgd_optimizer
from module import functional as F, OutputCollection, PartitionSpec


class LearnerTest(parameterized.TestCase):
    def assertNestedEqual(self, a, b):
        a_kv = utils.flatten_items(a)
        b_kv = utils.flatten_items(b)
        self.assertCountEqual([k for k, _ in a_kv], [k for k, _ in b_kv])
        a_dict = dict(a_kv)
        b_dict = dict(b_kv)
        for k in a_dict:
            np.testing.assert_array_equal(a_dict[k], b_dict[k], err_msg=k)

    @parameterized.parameters((0.1, 0), (0.1, 0.01))
    def testSGD(self, learning_rate, weight_decay):
        sgd = sgd_optimizer(learning_rate=learning_rate, weight_decay=weight_decay)
        params = jnp.asarray([0, 1, 2, -3], dtype=jnp.float32)
        state = sgd.init(params)

        def loss(x):
            return -jax.nn.log_softmax(x)[1]

        loss, grads = jax.value_and_grad(loss)(params)
        np.testing.assert_allclose(loss, 1.412078, atol=1e-6)
        np.testing.assert_allclose(
            grads, [0.089629, -0.756364, 0.662272, 0.004462], atol=1e-6
        )

        updates, updated_state = sgd.update(grads, state=state, params=params)
        np.testing.assert_allclose(
            updates, -learning_rate * (grads + weight_decay * params), atol=1e-6
        )

        updated_params = optax.apply_updates(params, updates)
        np.testing.assert_allclose(updated_params, params + updates, atol=1e-6)

    def testLearner(self):
        learning_rate = lambda step: 0.1 - 1e-4 * step
        weight_decay = 1e-4
        step = 0
        optimizer_cfg = config_for_function(sgd_optimizer).set(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
        learner: Learner = (
            Learner.default_config()
            .set(name="test", optimizer=optimizer_cfg)
            .instantiate(parent=None)
        )

        params = jnp.asarray([0, 1, 2, -3], dtype=jnp.float32)
        state = learner.init(model_params=params)

        def loss(x):
            return -jax.nn.log_softmax(x)[1]

        loss, grads = jax.value_and_grad(loss)(params)
        np.testing.assert_allclose(loss, 1.412078, atol=1e-6)
        np.testing.assert_allclose(
            grads, [0.089629, -0.756364, 0.662272, 0.004462], atol=1e-6
        )

        updated_params, output_collection = F(
            learner,
            method="update",
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(step=step, gradients=grads, model_params=params),
            output_collection_sections=[
                OutputCollection.SECTION_STATE_UPDATE,
                OutputCollection.SECTION_SUMMARY,
            ],
        )
        np.testing.assert_allclose(
            updated_params,
            params - learning_rate(step) * (grads + weight_decay * params),
            atol=1e-6,
        )
        self.assertCountEqual(
            [OutputCollection.SECTION_STATE_UPDATE, OutputCollection.SECTION_SUMMARY],
            output_collection.keys(),
        )
        summaries = output_collection[OutputCollection.SECTION_SUMMARY]
        self.assertEqual(
            {"learning_rate": learning_rate(step), "lr_schedule_step": 0}, summaries
        )
        state_updates = output_collection[OutputCollection.SECTION_STATE_UPDATE]
        self.assertNestedEqual(
            {
                "optimizer": (
                    optax.TraceState(trace=grads),
                    optax.EmptyState(),
                    optax.ScaleByScheduleState(count=1),
                )
            },
            state_updates,
        )

        self.assertEqual(
            LearnerState(
                optimizer=(
                    PartitionSpec(
                        ("model",),
                    ),
                    None,
                    None,
                )
            ),
            learner.create_state_partition_specs(PartitionSpec("model")),
        )


if __name__ == "__main__":
    absltest.main()

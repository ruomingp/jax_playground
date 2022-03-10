from functools import partial

import jax.nn
import numpy as np
import optax
from absl.testing import absltest
from jax import numpy as jnp

import schedule
from config import config_for_function
from learner import Learner, LearnerState
from module import ParameterPartitionSpec, PartitionSpec
from module import functional as F
from optimizer_base import OptParam, OptStatePartitionSpec
from optimizers import chain, clip_by_global_norm, sgd_optimizer
from test_utils import TestCase


class LearnerTest(TestCase):
    def testLearner(self):
        learning_rate = config_for_function(schedule.stepwise).set(
            sub=[0.1, 0.01, 0.001],
            start_step=[100, 200],
        )
        learning_rate_fn = schedule.as_schedule_fn(learning_rate)
        weight_decay = 1e-4
        step = 0
        sgd_cfg = config_for_function(sgd_optimizer).set(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
        optimizer_cfg = config_for_function(chain).set(
            args=(config_for_function(clip_by_global_norm), sgd_cfg),
        )
        learner: Learner = (
            Learner.default_config()
            .set(name="test", optimizer=optimizer_cfg)
            .instantiate(parent=None)
        )

        params = OptParam(
            value=jnp.asarray([0, 1, 2, -3], dtype=jnp.float32), factorization_spec=None
        )
        state = learner.init(model_params=params)

        def loss(x):
            return -jax.nn.log_softmax(x)[1]

        loss, grads = jax.value_and_grad(loss)(params.value)
        np.testing.assert_allclose(loss, 1.412078, atol=1e-6)
        np.testing.assert_allclose(grads, [0.089629, -0.756364, 0.662272, 0.004462], atol=1e-6)

        updated_params, output_collection = F(
            learner,
            method="update",
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(gradients=grads, model_params=params),
        )
        np.testing.assert_allclose(
            updated_params,
            params.value - learning_rate_fn(step) * (grads + weight_decay * params.value),
            atol=1e-6,
        )
        summaries = output_collection.summaries
        self.assertAlmostEqual(
            {
                "learning_rate": learning_rate_fn(step),
                "lr_schedule_step": 0,
                "gradient_norm": 1.0093285,
            },
            summaries,
        )
        state_updates = output_collection.state_updates
        self.assertNestedAllClose(
            {
                "optimizer": (
                    # clip_by_global_norm.
                    optax.EmptyState(),
                    # sgd.
                    (
                        optax.TraceState(trace=grads),
                        optax.EmptyState(),
                        optax.ScaleByScheduleState(count=jnp.ones([], dtype=jnp.int32)),
                    ),
                )
            },
            state_updates,
        )

        self.assertSequenceEqual(
            LearnerState(
                optimizer=(
                    # clip_by_global_norm.
                    None,
                    # sgd.
                    (
                        optax.TraceState(
                            trace=OptStatePartitionSpec(
                                shape=(4,), partition=PartitionSpec("model")
                            )
                        ),
                        OptStatePartitionSpec(shape=None, partition=None),
                        OptStatePartitionSpec(shape=None, partition=None),
                    ),
                )
            ),
            learner.create_state_partition_specs(
                ParameterPartitionSpec(
                    shape=(4,), partition=PartitionSpec("model"), factorization=None
                )
            ),
        )


if __name__ == "__main__":
    absltest.main()

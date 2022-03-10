import jax.nn
import optax
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp

import factorized_rms
from module import FactorizationSpec, ParameterPartitionSpec, PartitionSpec
from optimizer_base import OptParam
from optimizers import OptStatePartitionSpec, replicate
from test_utils import TestCase
from utils import flatten_items


class FactorizedRMSTest(TestCase):
    @parameterized.parameters(False, True)
    def testParity(self, factored):
        ref = replicate(optax.scale_by_factored_rms(factored=factored))
        exp = factorized_rms.scale_by_factored_rms(factored=factored)

        # Factorize 'w' but not 'b'.
        # By convention the largest dim is the "row", and the second largest is the "col".
        w_factorization = FactorizationSpec(axes=(None, "col", "row"))
        b_factorization = FactorizationSpec(axes=(None, None))

        # Partition on the 'expert' and 'model' axes.
        num_experts, model_dim, hidden_dim = 8, 150, 512
        parameter_partition_specs = dict(
            w=ParameterPartitionSpec(
                shape=[num_experts, model_dim, hidden_dim],
                partition=PartitionSpec("expert", None, "model"),
                factorization=w_factorization,
            ),
            b=ParameterPartitionSpec(
                shape=[num_experts, hidden_dim],
                partition=PartitionSpec("expert", "model"),
                factorization=b_factorization,
            ),
        )

        # The 'ref' optimizer is replicated.
        ref_partition = ref.partition(parameter_partition_specs)
        self.assertIsNone(ref_partition.partition)
        # The 'exp' optimizer is partitioned according to the partition of parameters and factorization spec.
        exp_partition = exp.partition(parameter_partition_specs)
        if factored:
            self.assertSequenceEqual(
                optax.FactoredState(
                    count=None,
                    v_row=dict(
                        b=None,
                        # 'v_row' does not have the 'row' dimension.
                        w=OptStatePartitionSpec(
                            shape=[num_experts, model_dim], partition=PartitionSpec("expert", None)
                        ),
                    ),
                    v_col=dict(
                        b=None,
                        # 'v_col' does not have the 'col' dimension.
                        w=OptStatePartitionSpec(
                            shape=[num_experts, hidden_dim],
                            partition=PartitionSpec("expert", "model"),
                        ),
                    ),
                    v=dict(
                        b=OptStatePartitionSpec(
                            shape=[num_experts, hidden_dim],
                            partition=PartitionSpec("expert", "model"),
                        ),
                        w=None,
                    ),
                ),
                exp_partition,
            )
        else:
            self.assertSequenceEqual(
                optax.FactoredState(
                    count=None,
                    v_row=dict(w=None, b=None),
                    v_col=dict(w=None, b=None),
                    v=jax.tree_map(
                        lambda pps: OptStatePartitionSpec(shape=pps.shape, partition=pps.partition),
                        parameter_partition_specs,
                    ),
                ),
                exp_partition,
            )

        # init() behaves the same between ref and exp.
        opt_params = dict(
            w=OptParam(
                value=jax.random.normal(
                    jax.random.PRNGKey(1), [num_experts, model_dim, hidden_dim]
                ),
                factorization_spec=w_factorization,
            ),
            b=OptParam(
                value=jnp.zeros([num_experts, hidden_dim]), factorization_spec=b_factorization
            ),
        )
        ref_opt_state = ref.init(opt_params)
        exp_opt_state = exp.init(opt_params)
        self.assertNestedAllClose(ref_opt_state, exp_opt_state)

        # Check exp_partition against exp_opt_state.
        state_spec_map = dict(flatten_items(exp_partition))
        for path, value in flatten_items(exp_opt_state):
            state_spec: OptStatePartitionSpec = state_spec_map.get(path)
            logging.info(
                "State: %s=%s(%s) state_spec=%s", path, value.dtype, value.shape, state_spec
            )
            if state_spec is None:
                self.assertEqual(value.size, 1, msg=f"{path}: {value.shape}")
                continue
            self.assertIsNotNone(state_spec, msg=f"{path}: {value.shape}")
            self.assertLen(state_spec.partition, len(value.shape))
            self.assertSequenceEqual(value.shape, state_spec.shape)
            for dim_size, dim_partition in zip(value.shape, state_spec.partition):
                if dim_partition is not None:
                    self.assertEqual(
                        dim_size % 8,
                        0,
                        msg=f"{path}: {dim_size} for {dim_partition} in {value.shape} vs. {state_spec}",
                    )

        # update() behaves the same between ref and exp.
        for step in range(10):
            updates = jax.tree_map(
                lambda x: jax.random.normal(jax.random.PRNGKey(100 + step), x.shape), opt_params
            )
            ref_scaled_updates, ref_opt_state = ref.update(updates, ref_opt_state, opt_params)
            exp_scaled_updates, exp_opt_state = exp.update(updates, exp_opt_state, opt_params)
            self.assertNestedAllClose(ref_opt_state, exp_opt_state)
            self.assertNestedAllClose(ref_scaled_updates, exp_scaled_updates)


if __name__ == "__main__":
    absltest.main()

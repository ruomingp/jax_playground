"""Optimization modules."""
from typing import NamedTuple

import jax
import optax
from absl import logging

import config as config_lib
from module import Module, NestedParameterPartitionSpec, NestedTensor
from optimizer_base import NestedOptParam, PartitionedGradientTransformation


class LearnerState(NamedTuple):
    optimizer: optax.OptState


class Learner(Module):
    """The learner module."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("optimizer", None, "The optimizer config.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self.optimizer: PartitionedGradientTransformation = cfg.optimizer.instantiate()
        if not isinstance(self.optimizer, PartitionedGradientTransformation):
            raise ValueError(
                f"optimizer must be a PartitionedGradientTransformation: {cfg.optimizer}"
            )

    def create_state_partition_specs(
        self, model_param_partition_specs: NestedParameterPartitionSpec
    ):
        return LearnerState(optimizer=self.optimizer.partition(model_param_partition_specs))

    def init(self, model_params: NestedOptParam) -> LearnerState:
        return LearnerState(optimizer=self.optimizer.init(model_params))

    def update(self, *, gradients: NestedTensor, model_params: NestedOptParam) -> NestedTensor:
        """Computes `model_params` updates with `gradients`."""
        parameter_updates, optimizer_state = self.optimizer.update(
            gradients, state=self.state.optimizer, params=model_params
        )
        self.add_state_update("optimizer", optimizer_state)
        updated_model_params = optax.apply_updates(
            jax.tree_map(lambda op: op.value, model_params), parameter_updates
        )
        return updated_model_params

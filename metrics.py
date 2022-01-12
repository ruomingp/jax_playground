import dataclasses
from absl import logging
from typing import Any, Dict

import jax
from jax import numpy as jnp

import config


@dataclasses.dataclass
@jax.tree_util.register_pytree_node_class
class WeightedScalar:
    mean: jnp.ndarray
    weight: jnp.ndarray

    def tree_flatten(self):
        return ((self.mean, self.weight), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __add__(self, other: "WeightedScalar"):
        weight = self.weight + other.weight
        if weight > 0:
            mean = (self.mean * self.weight + other.mean * other.weight) / weight
        else:
            mean = 0.0
        return WeightedScalar(mean, weight)


class MetricAccumulator(config.Configurable):
    def __init__(self, cfg: config.Config):
        super().__init__(cfg)
        self._scalars = {}

    def tree_map(self, *args, **kwargs):
        is_leaf = lambda x: isinstance(x, WeightedScalar)
        return jax.tree_map(*args, **kwargs, is_leaf=is_leaf)

    def update(self, input_batch: Any, model_outputs: Dict[str, Any]):
        logging.debug(
            "MetricAccumulator.update: current=%s update=%s",
            self._scalars,
            model_outputs,
        )
        scalars = self.tree_map(
            lambda x: x if isinstance(x, WeightedScalar) else tuple(), model_outputs
        )
        if not self._scalars:
            self._scalars = scalars
        else:
            self._scalars = self.tree_map(lambda x, y: x + y, self._scalars, scalars)
        logging.debug("MetricAccumulator.update: merged=%s", self._scalars)

    def summaries(self) -> Dict[str, Any]:
        return self.tree_map(lambda x: x.mean, self._scalars)

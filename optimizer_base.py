"""The optimizer API.

The API largely follows that of optax, but with a few changes to support partition and factorization, specifically:

1. The optimizer contains an additional "partition" function, which computes the partition specs for optimizer
   states for model parallelism.

2. init and update take params with OptParam, instead of jnp.ndarray, as leaf nodes. OptParam contains both the
   jnp.ndarray value and a FactorizationSpec, which allows optimizers that rely on tensor factorization to know
   which dimensions can be factorized.

   Note that we cannot derive factorization dims from only tensor shapes, since we do not want to factorize along
   the stacking axis for parameters of Repeat and Pipeline layers.
"""
import dataclasses
from typing import Any, Callable, Dict, NamedTuple, Sequence, Tuple, Union

import optax
import typing_extensions

from module import (
    FactorizationSpec,
    NestedParameterPartitionSpec,
    NestedPartitionSpec,
    PartitionSpec,
)
from utils import Tensor


@dataclasses.dataclass
class OptParam:
    """A parameter to be optimized by an optimizer."""

    value: Tensor
    factorization_spec: FactorizationSpec

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def shape(self):
        return self.value.shape


# NestedOptParam = Dict[str, Union[OptParam, "NestedOptParam"]]
NestedOptParam = Dict[str, Union[OptParam, Any]]

# Similar to optax.TransformInitFn, but with NestedOptParam as inputs so that factorization specs are available.
TransformInitFn = Callable[[NestedOptParam], optax.OptState]


class TransformUpdateFn(typing_extensions.Protocol):
    """Similar to optax.TransformUpdateFn, but with two differences:

    (1) params is required;
    (2) params is of type NestedOptParam and therefore contains factorization spec.
    """

    def __call__(
        self, updates: optax.Updates, state: optax.OptState, params: NestedOptParam
    ) -> Tuple[optax.Updates, optax.OptState]:
        ...


@dataclasses.dataclass
class OptStatePartitionSpec:
    shape: Sequence[int]
    partition: PartitionSpec


NestedOptStatePartitionSpec = Union[OptStatePartitionSpec, Dict, Sequence]

TransformPartitionSpecFn = Callable[[NestedParameterPartitionSpec], NestedOptStatePartitionSpec]


class PartitionedGradientTransformation(NamedTuple):
    init: TransformInitFn
    update: TransformUpdateFn
    partition: TransformPartitionSpecFn

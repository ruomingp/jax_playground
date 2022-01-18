import copy
from typing import Callable, NamedTuple, Tuple, Union

import optax

import config as config_lib
import schedule
from module import Module, NestedTensor, NestedPartitionSpec, Tensor, current_context

TransformPartitionSpecFn = Callable[[NestedPartitionSpec], NestedPartitionSpec]


class PartitionedGradientTransformation(NamedTuple):
    init: optax.TransformInitFn
    update: optax.TransformUpdateFn
    partition: TransformPartitionSpecFn


def chain(*elements):
    base = optax.chain(
        *[optax.GradientTransformation(init=e.init, update=e.update) for e in elements]
    )

    def partition(input_partition_spec):
        return tuple(e.partition(input_partition_spec) for e in elements)

    return PartitionedGradientTransformation(
        init=base.init, update=base.update, partition=partition
    )


def copy_partition(
    base: optax.GradientTransformation,
) -> PartitionedGradientTransformation:
    return PartitionedGradientTransformation(
        init=base.init,
        update=base.update,
        partition=lambda partition_spec: copy.deepcopy(partition_spec),
    )


def trace_partition(
    base: optax.GradientTransformation,
) -> PartitionedGradientTransformation:
    return PartitionedGradientTransformation(
        init=base.init,
        update=base.update,
        partition=lambda partition_spec: optax.TraceState(trace=partition_spec),
    )


def replicate(base: optax.GradientTransformation) -> PartitionedGradientTransformation:
    return PartitionedGradientTransformation(
        init=base.init, update=base.update, partition=lambda partition_spec: None
    )


def scale_from_learning_rate(learning_rate: schedule.Schedule):
    learning_rate_fn = schedule.as_schedule_fn(learning_rate)

    def scale(step):
        lr = learning_rate_fn(step)
        context = current_context()
        if context:
            context.add_summary("lr_schedule_step", step)
            context.add_summary("learning_rate", lr)
        return -lr

    return scale


def sgd_optimizer(
    learning_rate: schedule.Schedule,
    momentum: float = 0,
    weight_decay: float = 0,
) -> PartitionedGradientTransformation:
    return chain(
        trace_partition(optax.trace(decay=momentum)),
        replicate(optax.add_decayed_weights(weight_decay)),
        replicate(optax.scale_by_schedule(scale_from_learning_rate(learning_rate))),
    )


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

    def create_state_partition_specs(
        self, model_param_partition_specs: NestedPartitionSpec
    ):
        return LearnerState(
            optimizer=self.optimizer.partition(model_param_partition_specs)
        )

    def init(self, model_params: NestedTensor) -> LearnerState:
        return LearnerState(optimizer=self.optimizer.init(model_params))

    def update(
        self, *, step: Tensor, gradients: NestedTensor, model_params: NestedTensor
    ) -> NestedTensor:
        """Computes `model_params` updates with `gradients`."""
        parameter_updates, optimizer_state = self.optimizer.update(
            gradients, state=self.state.optimizer, params=model_params
        )
        self.add_state_update("optimizer", optimizer_state)
        updated_model_params = optax.apply_updates(model_params, parameter_updates)
        return updated_model_params

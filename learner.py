from typing import Callable, NamedTuple, Tuple, Union

import optax

import config as config_lib
import schedule
from module import Module, NestedTensor, NestedPartitionSpec, Tensor


def sgd_optimizer(
    learning_rate: Union[float, Callable[[int], float], config_lib.InstantiableConfig],
    momentum: float = 0,
    weight_decay: float = 0,
) -> optax.GradientTransformation:
    def scale(step):
        return -schedule.as_schedule_fn(learning_rate)(step)

    return optax.chain(
        optax.trace(decay=momentum),
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_schedule(scale),
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
        self.optimizer: optax.GradientTransformation = cfg.optimizer.instantiate()
        self.learning_rate: schedule.ScheduleFn = schedule.as_schedule_fn(
            cfg.optimizer.learning_rate
        )

    def create_state_partition_specs(
        self, model_param_partition_specs: NestedPartitionSpec
    ):
        cfg = self.config
        if cfg.optimizer.fn == optax.sgd:
            return LearnerState(
                optimizer=(optax.TraceState(trace=model_param_partition_specs), None)
            )
        raise NotImplementedError(cfg.optimizer)

    def init(self, model_params: NestedTensor) -> LearnerState:
        return LearnerState(optimizer=self.optimizer.init(model_params))

    def update(
        self, *, step: Tensor, gradients: NestedTensor, model_params: NestedTensor
    ) -> NestedTensor:
        """Computes `model_params` updates with `gradients`."""
        self.add_summary("learning_rate", self.learning_rate(step))
        parameter_updates, optimizer_state = self.optimizer.update(
            gradients, state=self.state.optimizer, params=model_params
        )
        self.add_state_update("optimizer", optimizer_state)
        parameter_updates = self._adjust_updates(
            parameter_updates, gradients=gradients, model_params=model_params
        )
        updated_model_params = optax.apply_updates(model_params, parameter_updates)
        return updated_model_params

    def _adjust_updates(
        self,
        updates: NestedTensor,
        *,
        gradients: NestedTensor,
        model_params: NestedTensor
    ) -> NestedTensor:
        # Subclasses can override this method to adjust the updates.
        del gradients, model_params
        return updates

from typing import NamedTuple, Tuple

import optax

import config
from module import Module, NestedTensor


class LearnerState(NamedTuple):
    optimizer: optax.OptState


class Learner(Module):
    """The learner module."""

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define("optimizer", None, "The optimizer config.")
        return cfg

    def __init__(self, cfg: config.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        self.optimizer: optax.GradientTransformation = cfg.optimizer.instantiate()

    def init(self, model_params: NestedTensor) -> LearnerState:
        return LearnerState(optimizer=self.optimizer.init(model_params))

    def update(self, state: LearnerState, *, gradients: NestedTensor, model_params: NestedTensor) -> Tuple[
        LearnerState, NestedTensor]:
        parameter_updates, optimizer_state = self.optimizer.update(
            gradients, state=state.optimizer, params=model_params)
        parameter_updates = self._adjust_updates(parameter_updates, gradients=gradients, model_params=model_params)
        updated_model_params = optax.apply_updates(model_params, parameter_updates)
        return LearnerState(optimizer=optimizer_state), updated_model_params

    def _adjust_updates(self, updates: NestedTensor, *, gradients: NestedTensor,
                        model_params: NestedTensor) -> NestedTensor:
        # Subclasses can override this method to adjust the updates.
        del gradients, model_params
        return updates

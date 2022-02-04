import math
from typing import Callable, List, Union

import jax
from jax import numpy as jnp

import config as config_lib
from utils import Tensor

ScheduleFn = Callable[[Tensor], Tensor]
Schedule = Union[float, Tensor, ScheduleFn, config_lib.InstantiableConfig]


def as_schedule_fn(s: Schedule) -> ScheduleFn:
    if isinstance(s, (config_lib.InstantiableConfig, config_lib.FunctionConfig)):
        return s.instantiate()
    if isinstance(s, float):
        return lambda step: float(s)
    assert callable(s), s
    return s


def polynomial(
    *,
    begin_step: int = 0,
    begin_value: float = 0,
    end_step: int = 1,
    end_value: float = 0,
    power: float = 1,
) -> ScheduleFn:
    """A polynomial (linear when power=1) schedule.

    Args:
        begin_step: The first step of polynomial schedule.
        begin_value: The begin value of polynomial schedule.
        end_step: The end step of polynomial schedule. Must be > begin_step.
        end_value: The end value of polynomial schedule.
        power: The polynomial power.

    Returns:
        A ScheduleFn according to the spec.
    """
    if begin_step >= end_step:
        raise ValueError(f"begin_step {begin_step} must be < end_step {end_step}.")

    def fn(step: Tensor) -> Tensor:
        frac = (step - begin_step) / (end_step - begin_step)
        frac = jnp.minimum(1.0, jnp.maximum(0.0, frac))
        return begin_value + (frac**power) * (end_value - begin_value)

    return fn


def exponential(
    *,
    begin_step: int = 0,
    begin_value: float = 0,
    end_step: int = 1,
    end_value: float = 0,
) -> ScheduleFn:
    """An exponential schedule.

    Args:
        begin_step: The first step of the schedule.
        begin_value: The begin value of the schedule.
        end_step: The end step of the schedule. Must be > begin_step.
        end_value: The end value of the schedule.

    Returns:
        A ScheduleFn according to the spec.
    """
    if begin_step >= end_step:
        raise ValueError(f"begin_step {begin_step} must be < end_step {end_step}.")
    if begin_value <= 0 or end_value <= 0:
        raise ValueError(
            f"begin_value ({begin_value}) and end_value ({end_value}) must be both positive."
        )

    log_fn = polynomial(
        begin_step=begin_step,
        begin_value=math.log(begin_value),
        end_step=end_step,
        end_value=math.log(end_value),
    )

    def fn(step: Tensor) -> Tensor:
        return math.exp(log_fn(step))

    return fn


def inverse_sqrt(step: int) -> float:
    return jnp.maximum(1, step) ** -0.5


def stepwise(sub: List[Schedule], start_step: List[int]) -> ScheduleFn:
    """A composite schedule consisting of multiple sub-schedules.

    The first sub-schedule starts at step 0. For the rest of sub-schedules, sub[i] starts at start_step[i-1].

    The step passed to sub-schedule is the relative step from its start step, so that the values of each sub-schedule
    do not depend on other sub-schedules.

    Args:
        sub: a sequence of N sub-schedules.
        start_step: a sequence of N-1 integers. start_steps[i] represents the starting step of sub[i+1].
            0 <= start_step[i] <= start_step[i+1].

    Returns:
        A composite schedule.
    """
    if len(sub) != len(start_step) + 1:
        raise ValueError(f"Unexpected length: {len(sub)} != {len(start_step)} + 1")
    if not all(step >= 0 for step in start_step):
        raise ValueError(f"start_step must be >= 0: {start_step}")
    sub = [as_schedule_fn(s) for s in sub]
    all_start_steps = [0] + start_step
    all_limit_steps = start_step + [-1]

    def fn(step: Tensor) -> Tensor:
        values = [s(jnp.maximum(0, step - start)) for s, start in zip(sub, all_start_steps)]
        activations = [
            jnp.logical_and(
                jax.lax.le(start, step),
                jnp.logical_or(limit < 0, jax.lax.lt(step, limit)),
            ).astype(jnp.float32)
            for start, limit in zip(all_start_steps, all_limit_steps)
        ]
        return sum([value * activation for value, activation in zip(values, activations)])

    return fn

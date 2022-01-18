import math
import numbers
from typing import Callable, Sequence, Union

import config as config_lib

ScheduleFn = Callable[[int], float]
Schedule = Union[float, ScheduleFn, config_lib.InstantiableConfig]


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

    def fn(step: int) -> float:
        frac = (step - begin_step) / (end_step - begin_step)
        frac = min(1.0, max(0.0, frac))
        return begin_value + (frac ** power) * (end_value - begin_value)

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

    def fn(step: int) -> float:
        return math.exp(log_fn(step))

    return fn


def inverse_sqrt(step: int) -> float:
    return max(1, step) ** -0.5


def stepwise(sub: Sequence[Schedule], start_step: Sequence[int]) -> ScheduleFn:
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
    sub = [as_schedule_fn(s) for s in sub]

    def fn(step: int):
        index = 0
        start = 0
        while index < len(start_step):
            if step < start_step[index]:
                break
            start = start_step[index]
            index += 1
        return sub[index](step - start)

    return fn

"""Base class of modules.

Design choices:
* All module hyper-parameters are encapsulated by the module's config.
* Every module has a name, default dtype, and an optional param_init config.
* Every module has a parent except the root module.
* A module's config is frozen upon __init__. This prevents the config from being modified by accident.
* Module.config returns a copy of the module's config. This allows the caller to make changes without affecting the
  original config.
"""
import contextlib
import copy
import dataclasses
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import jax
from absl import logging
from jax import numpy as jnp
from jax.experimental.pjit import PartitionSpec

import config as config_lib
import param_init
from metrics import WeightedScalar
from utils import NestedTensor, Tensor

# NestedPartitionSpec = Optional[Union[PartitionSpec, Dict[str, "NestedPartitionSpec"]]]
NestedPartitionSpec = Optional[Union[PartitionSpec, Dict[str, Any]]]


class OutputCollection(NamedTuple):
    summaries: NestedTensor
    state_updates: NestedTensor

    def add_child(self, name: str) -> "OutputCollection":
        if not re.fullmatch("^[a-z][a-z0-9_]*$", name):
            raise ValueError(f'Invalid child name "{name}"')
        if name in self.summaries or name in self.state_updates:
            raise ValueError(f"{name} already present")
        child = new_output_collection()
        self.summaries[name] = child.summaries
        self.state_updates[name] = child.state_updates
        return child


def new_output_collection():
    return OutputCollection(summaries={}, state_updates={})


@dataclass
class InvocationContext:
    module: "Module"
    # The state of the module.
    state: NestedTensor
    is_training: bool
    prng_key: jax.random.KeyArray
    output_collection: OutputCollection

    def clone(self, **override_kwargs):
        kwargs = {}
        for field in dataclasses.fields(self):
            k = field.name
            if k in override_kwargs:
                kwargs[k] = override_kwargs[k]
            elif k in ("module", "is_training", "prng_key"):
                kwargs[k] = getattr(self, k)
            else:
                kwargs[k] = copy.deepcopy(getattr(self, k))
        assert kwargs["module"] is self.module
        return InvocationContext(**kwargs)

    def add_child(self, name: str, *, state=None) -> "InvocationContext":
        self.prng_key, child_key = jax.random.split(self.prng_key)
        state = state or self.state.get(name)
        return InvocationContext(
            module=getattr(self.module, name),
            is_training=self.is_training,
            prng_key=child_key,
            state=state,
            output_collection=self.output_collection.add_child(name),
        )

    def add_summary(self, name: str, value: Union[WeightedScalar, Tensor]):
        self.output_collection.summaries[name] = value

    def add_state_update(self, name: str, value: Tensor):
        self.output_collection.state_updates[name] = value

    def get_summaries(self):
        return self.output_collection.summaries

    def get_state_updates(self):
        return self.output_collection.state_updates


@dataclass
class ContextStack(threading.local):
    stack: List[InvocationContext]


_global_context_stack = ContextStack(stack=[])


def current_context() -> Optional[InvocationContext]:
    global _global_context_stack
    if not _global_context_stack.stack:
        return None
    return _global_context_stack.stack[-1]


@contextlib.contextmanager
def set_current_context(context: InvocationContext):
    global _global_context_stack
    try:
        _global_context_stack.stack.append(context)
        yield context
    finally:
        _global_context_stack.stack.pop(-1)


@contextlib.contextmanager
def child_context(name: str, **kwargs):
    context = current_context().add_child(name, **kwargs)
    with set_current_context(context) as c:
        yield c


class Module(config_lib.Configurable):
    @classmethod
    def default_config(cls):
        cfg = config_lib.InstantiableConfig(cls)
        cfg.define("name", "", "Name of this module.")
        cfg.define("vlog", 0, "The maximum vlog level.")
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional["Module"]):
        super().__init__(cfg)
        if not cfg.name:
            raise ValueError(f"Module must have a name: {cfg.debug_string()}")
        self._parent = parent  # avoid adding parent to self._modules
        self._children: Dict[str, "Module"] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            try:
                return self.__dict__[name]
            except KeyError:
                raise AttributeError(name)
        else:
            try:
                return self._children[name]
            except KeyError:
                raise AttributeError(f"{type(self)}.{name}")

    @property
    def parent(self):
        return self._parent

    def path(self):
        if self.parent is None:
            return self.config.name
        return f"{self.parent.path()}.{self.config.name}"

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"{type(self)}@{self.path()}"

    def vlog(self, level, msg, *args, **kwargs):
        if level <= self.config.vlog:
            logging.info(msg, *args, **kwargs)

    def _add_child(self, name: str, child_config: config_lib.Config, **kwargs) -> "Module":
        if not re.fullmatch("^[a-z][a-z0-9_]*$", name):
            raise ValueError(f'Invalid child name "{name}"')
        child_config = copy.deepcopy(child_config)
        if not issubclass(child_config.cls, Module):
            raise TypeError(f"add_child expects a Module config, got {child_config}")
        if child_config.name and child_config.name != name:
            raise ValueError(
                f"child_config already has a different name: {child_config.name} vs. {name}"
            )
        child_config.name = name
        module = child_config.freeze().instantiate(parent=self, **kwargs)
        if name in self._children:
            raise ValueError(f"Child {name} already exists")
        self._children[name] = module
        return module

    @property
    def children(self):
        return self._children

    def get_invocation_context(self):
        context = current_context()
        if not context:
            raise RuntimeError("Module invocation context not found")
        if context.module is not self:
            raise RuntimeError(f"Module mismatch: {context.module} vs. {self}")
        return context

    @property
    def is_training(self) -> bool:
        return self.get_invocation_context().is_training

    @property
    def prng_key(self) -> jax.random.KeyArray:
        return self.get_invocation_context().prng_key

    @property
    def state(self):
        return self.get_invocation_context().state

    def add_summary(self, name: str, value: Union[WeightedScalar, Tensor]):
        return self.get_invocation_context().add_summary(name, value)

    def add_state_update(self, name: str, value: Tensor):
        return self.get_invocation_context().add_state_update(name, value)

    def __call__(self, *args, method="forward", context=None, **kwargs) -> Any:
        if len(args) > 1:
            logging.log_first_n(
                logging.WARNING,
                "Multiple positional arguments for %s.%s. Consider using keyword arguments instead.",
                3,
                type(self),
                method,
            )

        def f():
            return getattr(self, method)(*args, **kwargs)

        if context is not None:
            with set_current_context(context):
                return f()
        context = current_context()
        if context is None:
            raise ValueError(
                f"context is required when {self} is invoked outside of an InvocationContext. "
                "Consider using module.functional() to wrap the call."
            )
        if context.module is self:
            return f()
        if context.module is self.parent:
            with child_context(self.config.name):
                return f()
        raise ValueError("context.module does not match self")


def functional(
    module: Module,
    prng_key: jax.random.KeyArray,
    state: NestedTensor,
    inputs: Union[Sequence[Any], Dict[str, Any]],
    *,
    method: str = "forward",
    is_training: bool,
) -> Tuple[Any, OutputCollection]:
    """Invokes <module>.<method> in a pure functional fashion.

    The invocation will not depend on external inputs or have any side effects. The results only depend on the given
    inputs. All outputs are reflected in the return value.

    TODO(ruoming): support output collection filter.

    Args:
        module: The Module to invoke.
        prng_key: the pseudo-random number generate key.
        state: The input state of the module, including model parameters if the module contains a model.
        inputs: The inputs for the method. If it's a sequence, it represents the positional args. If it's a dict,
          it represents keyword args.
        method: The Module method to invoke.
        is_training: Whether the invocation should run in the training mode.

    Returns:
        (method_outputs, output_collection), where
        - method_outputs are the return value of the method.
        - output_collection is an OutputCollection containing summaries and state updates.
    """
    context = InvocationContext(
        module=module,
        state=state,
        output_collection=new_output_collection(),
        is_training=is_training,
        prng_key=prng_key,
    )

    with set_current_context(context):
        if isinstance(inputs, dict):
            input_args, input_kwargs = [], inputs
        else:
            input_args, input_kwargs = inputs, {}
        method_outputs = getattr(module, method)(*input_args, **input_kwargs)
    return method_outputs, context.output_collection


@dataclasses.dataclass
class ParameterSpec:
    shape: Sequence[int]
    # If None, the parameter will not be partitioned and will be replicated.
    # If a sequence, it should have at most len(shape) elements. partition_spec[i] describes partitioning for shape[i],
    # where each value can be:
    # - None: do not partition along this axis, or
    # - 'model': partition along this axis across the 'model' dim of the device mesh.
    partition_spec: Optional[Sequence[Optional[str]]]
    # The data type of the param. If None, uses the layer's default dtype.
    dtype: Optional[jnp.dtype] = None
    # The initializer of the param. If None, uses the layer's default initializer.
    initializer: Optional[param_init.Initializer] = None


# When pytype supports recursive typing:
# NestedParameterSpec = Dict[str, Union[ParameterSpec, "NestedParameterSpec"]]
NestedParameterSpec = Dict[str, Union[ParameterSpec, Any]]


class BaseLayer(Module):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.define(
            "dtype",
            None,
            "If not None, the default parameter dtype. "
            "If None, inherits from the parent module.",
        )
        cfg.define(
            "param_init",
            None,
            "If not None, parameter initialization config of this module. "
            "If None, inherits from the parent module.",
        )
        cfg.define(
            "param_partition_spec",
            None,
            "The partition spec for the layer parameters. "
            "When the layer contains a weight parameter and a bias parameter, "
            "the partition spec will be defined in terms of the weight parameter, "
            "while the partition spec of the bias parameter can be derived accordingly.",
        )
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional["Module"]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.param_init is not None:
            init = cfg.param_init.instantiate()
        elif parent is None:
            init = param_init.DefaultInitializer.default_config().instantiate()
        else:
            init = None
        self._param_init = init

    def dtype(self):
        if self.config.dtype is not None:
            return self.config.dtype
        if self.parent is not None:
            return self.parent.dtype()
        return jnp.float32

    def param_init(self) -> param_init.Initializer:
        init = getattr(self, "_param_init", None)
        if init is not None:
            return init
        return self.parent.param_init()

    def create_parameter_specs_recursively(self) -> NestedParameterSpec:
        specs = self._create_layer_parameter_specs()
        for name, spec in specs.items():
            if spec.dtype is None:
                spec.dtype = self.dtype()
            if spec.initializer is None:
                spec.initializer = self.param_init()
        for name, child in self._children.items():
            assert name not in specs
            specs[name] = child.create_parameter_specs_recursively()
        return specs

    def _create_layer_parameter_specs(self) -> Dict[str, ParameterSpec]:
        """Subclasses can override this method to add layer parameters."""
        return {}

    def initialize_parameters_recursively(
        self,
        prng_key: jax.random.KeyArray,
        param_specs: Optional[NestedParameterSpec] = None,
    ) -> NestedTensor:
        if param_specs is None:
            param_specs = self.create_parameter_specs_recursively()
        params = {}
        for name, child in param_specs.items():
            prng_key, child_key = jax.random.split(prng_key)
            if isinstance(child, ParameterSpec):
                params[name] = self._initialize_parameter(
                    name, prng_key=child_key, parameter_spec=child
                )
            else:
                params[name] = self.initialize_parameters_recursively(
                    prng_key=child_key, param_specs=child
                )
        return params

    def _initialize_parameter(
        self, name: str, *, prng_key: jax.random.KeyArray, parameter_spec: ParameterSpec
    ) -> Tensor:
        """Adds a parameter with the given name and shape.

        Args:
            name: The parameter name.
            prng_key: The pseudo random generator key.
            parameter_spec: The parameter specification.

        Returns:
            The created parameter.
        """
        if name in self._children:
            raise ValueError(f"Child {name} already exists.")
        return parameter_spec.initializer.initialize(
            name,
            prng_key=prng_key,
            shape=parameter_spec.shape,
            dtype=parameter_spec.dtype,
        )

    @property
    def parameters(self):
        return self.state

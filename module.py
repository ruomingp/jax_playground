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
from collections import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Set

import jax
from absl import logging
from jax import numpy as jnp
from jax.experimental.pjit import PartitionSpec

import config as config_lib
import param_init
from utils import NestedTensor, Tensor


NestedPartitionSpec = Dict[str, Union[PartitionSpec, "NestedPartitionSpec"]]


class FrozenDict(abc.Mapping):
    def __init__(self, *args, **kwargs):
        self.contents = dict(*args, **kwargs)

    def __iter__(self):
        return iter(self.contents)

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, name):
        return self.contents[name]

    def __eq__(self, other):
        return isinstance(other, FrozenDict) and self.contents == other.contents

    def __hash__(self):
        return hash(tuple(self.contents.items()))

    def __repr__(self):
        return f"FrozenDict({self.contents})"


class OutputCollection:
    # Some standard section names.
    SECTION_DEFAULT = ""
    SECTION_SUMMARY = "summary"
    SECTION_PARAMETER_UPDATE = "parameter_update"

    def __init__(self):
        self._all_names: Set[str] = set()
        self._children: Dict[str, "OutputCollection"] = dict()
        self._sections: Dict[str, Dict[str, jnp.ndarray]] = dict()

    def _check_name(self, name):
        if not re.fullmatch("^[a-z][a-z0-9_]*$", name):
            raise ValueError(f'Invalid name "{name}"')
        if name in self._all_names:
            raise ValueError(f"{name} already added as a child or output")
        self._all_names.add(name)

    def add_value(self, name: str, value: Any, *, section: str = SECTION_DEFAULT):
        self._check_name(name)
        if section not in self._sections:
            self._sections[section] = {}
        self._sections[section][name] = value

    def add_child(self, name: str) -> "OutputCollection":
        self._check_name(name)
        self._children[name] = OutputCollection()
        return self._children[name]

    def get_values_recursively(
        self, section: str = SECTION_DEFAULT
    ) -> Dict[str, Union[jnp.ndarray, dict]]:
        results = {}
        if section in self._sections:
            results.update(self._sections[section])
        for child_name, child_collection in self._children.items():
            child_values = child_collection.get_values_recursively(section)
            if child_values:
                results[child_name] = child_values
        return results


@dataclass
class InvocationContext:
    module: "Module"
    parameters: NestedTensor
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

    def add_child(self, name: str, *, parameters=None) -> "InvocationContext":
        self.prng_key, child_key = jax.random.split(self.prng_key)
        parameters = parameters or self.parameters.get(name)
        return InvocationContext(
            module=getattr(self.module, name),
            is_training=self.is_training,
            prng_key=child_key,
            parameters=parameters,
            output_collection=self.output_collection.add_child(name),
        )

    def get_summaries(self):
        return self.output_collection.get_values_recursively(
            OutputCollection.SECTION_SUMMARY
        )

    def get_parameter_updates(self):
        return self.output_collection.get_values_recursively(
            OutputCollection.SECTION_PARAMETER_UPDATE
        )


@dataclass
class ContextStack(threading.local):
    stack: List[InvocationContext]


_global_context_stack = ContextStack(stack=[])


def current_context() -> InvocationContext:
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
def root_context(context: InvocationContext):
    global _global_context_stack
    if _global_context_stack.stack:
        raise ValueError("Already within a InvocationContext")
    with set_current_context(context) as c:
        yield c


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

    def _add_child(
        self, name: str, child_config: config_lib.Config, **kwargs
    ) -> "Module":
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

    def make_invocation_context(self, **kwargs):
        return InvocationContext(
            module=self, output_collection=OutputCollection(), **kwargs
        )

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
    def parameters(self):
        return self.get_invocation_context().parameters

    def add_summary(self, name: str, value: jnp.ndarray):
        return self.get_invocation_context().output_collection.add_value(
            name, value, section=OutputCollection.SECTION_SUMMARY
        )

    def add_parameter_update(self, name: str, value: jnp.ndarray):
        return self.get_invocation_context().output_collection.add_value(
            name, value, section=OutputCollection.SECTION_PARAMETER_UPDATE
        )

    def __call__(self, *args, method="forward", context=None, **kwargs) -> Any:
        if len(args) > 1:
            logging.log_first_n(
                logging.WARNING,
                "Multiple positional arguments for %s.%s. Consider using keyword arguments instead.",
                3,
                type(self),
                method,
            )

        f = lambda: getattr(self, method)(*args, **kwargs)
        if context is not None:
            with set_current_context(context):
                return f()
        context = current_context()
        if context is None:
            raise ValueError(
                f"context is required when {self} is invoked outside of an InvocationContext."
            )
        if context.module is self:
            return f()
        if context.module is self.parent:
            with child_context(self.config.name):
                return f()
        raise ValueError("context.module does not match self")


@dataclasses.dataclass
class ParameterSpec:
    shape: Sequence[int]
    partition_spec: PartitionSpec
    # The data type of the param. If None, uses the layer's default dtype.
    dtype: Optional[jnp.dtype] = None
    # The initializer of the param. If None, uses the layer's default initializer.
    initializer: Optional[param_init.Initializer] = None


NestedParameterSpec = Dict[str, Union[ParameterSpec, "NestedParameterSpec"]]


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

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
from typing import Any, Dict, List, Optional, Sequence, Union

import jax
from absl import logging
from jax import numpy as jnp

import config as config_lib
import param_init

NestedParameters = Dict[str, Union[jnp.ndarray, 'NestedParameters']]


class OutputCollection:

    # Some standard collection names.
    SUMMARY = 'summary'
    PARAMETER_UPDATES = 'parameter_updates'

    def __init__(self, outputs: Dict[str, Union[jnp.ndarray, 'OutputCollection']]):
        self._outputs = outputs

    def add_value(self, name: str, value: jnp.ndarray):
        self._insert(name, value)

    def add_child(self, name: str) -> 'OutputCollection':
        child = OutputCollection({})
        self._insert(name, child)
        return child

    def _insert(self, name: str, value: Union[jnp.ndarray, 'OutputCollection']):
        if name in self._outputs:
            raise ValueError(f'{name} already added')
        self._outputs[name] = value


def standard_output_collections():
    return {name: OutputCollection({}) for name in (OutputCollection.SUMMARY, OutputCollection.PARAMETER_UPDATES)}


@dataclass
class InvocationContext:
    module: 'Module'
    parameters: NestedParameters
    is_training: bool
    prng_key: jax.random.KeyArray
    output_collections: Optional[Dict[str, OutputCollection]] = None

    def clone(self, **override_kwargs):
        kwargs = {}
        for field in dataclasses.fields(self):
            k = field.name
            if k in override_kwargs:
                kwargs[k] = override_kwargs[k]
            elif k in ('module', 'is_training', 'prng_key'):
                kwargs[k] = getattr(self, k)
            else:
                kwargs[k] = copy.deepcopy(getattr(self, k))
        assert kwargs['module'] is self.module
        return InvocationContext(**kwargs)

    def add_child(self, name: str) -> 'InvocationContext':
        self.prng_key, child_key = jax.random.split(self.prng_key)
        if self.output_collections is None:
            child_output_collections = None
        else:
            child_output_collections = {
                collection_name: collection.add_child(name) for collection_name, collection in
                self.output_collections.items()
            }
        return InvocationContext(is_training=self.is_training, prng_key=child_key, parameters=self.parameters.get(name),
                                 output_collections=child_output_collections)


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
        raise ValueError('Already within a InvocationContext')
    with set_current_context(context) as c:
        yield c


@contextlib.contextmanager
def child_context(name: str):
    context = current_context().add_child(name)
    with set_current_context(context) as c:
        yield c


class Module(config_lib.Configurable):

    @classmethod
    def default_config(cls):
        cfg = config_lib.InstantiableConfig(cls)
        cfg.define('name', '', 'Name of this module.')
        cfg.define('dtype', None,
                   'If not None, the default parameter dtype. '
                   'If None, inherits from the parent module.')
        cfg.define('param_init', None,
                   'If not None, parameter initialization config of this module. '
                   'If None, inherits from the parent module.')
        return cfg

    def __init__(self, cfg: config_lib.Config, *, parent: Optional['Module']):
        super().__init__(cfg)
        if not cfg.name:
            raise ValueError(f'Module must have a name: {cfg.debug_string()}')
        self._parent = parent  # avoid adding parent to self._modules
        cfg = self.config
        if cfg.param_init is not None:
            init = cfg.param_init.instantiate()
        elif parent is None:
            init = param_init.DefaultInit.default_config().instantiate()
        else:
            init = None
        self._param_init = init
        self._children: Dict[str, 'Module'] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            try:
                return self.__dict__[name]
            except KeyError:
                raise AttributeError(name)
        else:
            try:
                return self._children[name]
            except KeyError:
                raise AttributeError(name)

    @property
    def parent(self):
        return self._parent

    def path(self):
        if self.parent is None:
            return self.config.name
        return f'{self.parent.path()}.{self.config.name}'

    def __str__(self):
        return f'{type(self)}@{self.path()}'

    def dtype(self):
        if self.config.dtype is not None:
            return self.config.dtype
        if self.parent is not None:
            return self.parent.dtype()
        return jnp.float32

    def param_init(self) -> param_init.Init:
        init = getattr(self, '_param_init', None)
        if init is not None:
            return init
        return self.parent.param_init()

    def _add_child(self, name: str, child_config: config_lib.Config, **kwargs) -> 'Module':
        if not re.fullmatch('^[a-z][a-z0-9_]*$', name):
            raise ValueError(f'Invalid child name "{name}"')
        child_config = copy.deepcopy(child_config)
        if not issubclass(child_config.cls, Module):
            raise TypeError(f'add_child expects a Module config, got {child_config}')
        if child_config.name is not None and child_config.name != name:
            raise ValueError(f'child_config already has a different name: {child_config.name} vs. {name}')
        child_config.name = name
        module = child_config.freeze().instantiate(parent=self, **kwargs)
        if name in self._children:
            raise ValueError(f'Child {name} already exists')
        self._children[name] = module
        return module

    def initialize_parameters_recursively(self, prng_key: jax.random.KeyArray) -> NestedParameters:
        prng_key, module_key = jax.random.split(prng_key)
        params = self._initialize_module_parameters(prng_key=module_key)
        for name, child in self._children.items():
            assert name not in params
            prng_key, child_key = jax.random.split(prng_key)
            params[name] = child.initialize_parameters_recursively(prng_key=child_key)
        return params

    def _initialize_module_parameters(self, *, prng_key: jax.random.KeyArray) -> NestedParameters:
        raise NotImplementedError(type(self))

    def _initialize_parameter(self, name: str, *, prng_key: jax.random.KeyArray, shape: Sequence[int],
                              dtype: Optional[jnp.dtype] = None) -> jnp.ndarray:
        """Adds a parameter with the given name and shape.

        **kwargs will be passed to torch.emtpy(). If 'dtype' is not in kwargs, self.dtype() will be used.

        Args:
            name: The parameter name.
            prng_key: The pseudo random generator key.
            shape: The parameter shape.
            dtype: The parameter data type.

        Returns:
            The created parameter.
        """
        if name in self._children:
            raise ValueError(f'Child {name} already exists.')
        return self.param_init().initialize(name, prng_key=prng_key, shape=shape, dtype=dtype or self.dtype)

    def make_invocation_context(self, **kwargs):
        return InvocationContext(module=self, **kwargs)

    def get_invocation_context(self):
        context = current_context()
        if not context:
            raise RuntimeError('Module invocation context not found')
        if context.module is not self:
            raise RuntimeError(f'Module mismatch: {context.module} vs. {self}')
        return context

    @property
    def is_training(self) -> bool:
        return self.get_invocation_context().is_training

    @property
    def parameters(self):
        return self.get_invocation_context().parameters

    def __call__(self, *args, method='forward', context=None, **kwargs) -> Any:
        if len(args) > 1:
            logging.log_first_n(logging.WARNING,
                                'Multiple positional arguments for %s.%s. Consider using keyword arguments instead.',
                                3, type(self), method)

        f = lambda: getattr(self, method)(*args, **kwargs)
        if context is not None:
            with set_current_context(context):
                return f()
        context = current_context()
        if context is None:
            raise ValueError(f'context is required when {self} is invoked outside of an InvocationContext.')
        if context.module is self:
            return f()
        if context.module is self.parent:
            with child_context(self.config.name):
                return f()
        raise ValueError('context.module does not match self')

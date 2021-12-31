"""Classes to represent configs for ML layers, inputs, and models.

Adapted from https://github.com/tensorflow/lingvo/blob/master/lingvo/core/hyperparams.py.

Example usage for configuring a module:

    class MyModule(Configurable):

        @classmethod
        def default_config(cls):
            cfg = super().default_config()
            cfg.define('num_layers', 0, 'Number of layers.')
            cfg.define('input_dim', 0, 'Input dimension.')
            cfg.define('output_dim', 0, 'Output dimension.')
            return cfg

        def __init__(self, cfg: Config):
            super().__init__(cfg)
            cfg = self.config
            for layer_i in range(cfg.num_layers):
                ...

    def create_foo_module():
        cfg = MyModule.default_config()
        cfg.num_layers = 8
        cfg.set(input_dim=512, output_dim=256)
        return cfg.instantiate()

Config can also be used for third-party classes with InstantiableConfig.for_class(), for example:

    cfg.define('input_projection', InstantiableConfig.for_class(nn.Linear),
               'The linear layer for input projection')
"""

import copy
import dataclasses
import enum
import inspect
import re
from typing import Any, Callable, List, Optional, Tuple

import numpy as np


class InvalidConfigNameError(ValueError):
    pass


class InvalidConfigValueError(TypeError):
    pass


class FieldAlreadyExistsError(ValueError):
    pass


class UnknownFieldError(AttributeError):
    pass


class FrozenConfigError(RuntimeError):
    pass


def validate_config_field_name(name: str) -> None:
    if not re.fullmatch('^[a-z][a-z0-9_]*$', name):
        raise InvalidConfigNameError(f'Invalid config field name "{name}"')


def validate_config_field_value(value: Any) -> None:
    if isinstance(value, (list, tuple)):
        for x in value:
            validate_config_field_value(x)
    elif isinstance(value, dict):
        for _, v in value.items():
            validate_config_field_value(v)
    elif dataclasses.is_dataclass(value):
        validate_config_field_value(value.__dict__)
    elif value is None or isinstance(value, (Config, type, int, float, str, enum.Enum, np.dtype)):
        pass
    else:
        raise InvalidConfigValueError(f'Invalid config value type {type(value)} for value "{value}"')


def _is_named_tuple(x):
    """Returns whether an object is an instance of a collections.namedtuple.

    Examples::
      _is_named_tuple((42, 'hi')) ==> False
      Foo = collections.namedtuple('Foo', ['a', 'b'])
      _is_named_tuple(Foo(a=42, b='hi')) ==> True

    Args:
      x: The object to check.
    """
    return isinstance(x, tuple) and hasattr(x, '_fields') and hasattr(x, '_asdict')


class _ConfigField:

    def __init__(self, name, default_value, description):
        validate_config_field_name(name)
        validate_config_field_value(default_value)
        self._name = name
        self._value = default_value
        self._default_value = default_value
        self._description = description

    @property
    def value(self):
        return self._value

    @property
    def default_value(self):
        return self._default_value

    @property
    def description(self):
        return self._description

    def set_value(self, value):
        validate_config_field_value(value)
        self._value = value


class Config:
    """A Config consists of a dict of {name: field}."""

    def __init__(self):
        self._fields = {}  # {str: _ConfigField}
        self._frozen = False

    def freeze(self) -> 'Config':
        self._frozen = True

        def _enter(key: str, val: Any, default_sub_key_vals: List):
            if isinstance(val, Config):
                val._frozen = True
            return default_sub_key_vals

        self.visit(lambda k, v: None, enter_fn=_enter)
        return self

    def define(self, name: str, default_value: Any, description: str) -> _ConfigField:
        if name in self._fields:
            raise FieldAlreadyExistsError(f'Config field {name} already exists')
        field = _ConfigField(name, default_value, description)
        self._fields[name] = field
        return field

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            self.__dict__[name] = value
        else:
            if self._frozen:
                raise FrozenConfigError(f'Trying to modify a frozen config: {name}={value}')
            try:
                self._fields[name].set_value(value)
            except KeyError:
                raise UnknownFieldError(self._key_error_string(name))

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            try:
                return self.__dict__[name]
            except KeyError:
                raise AttributeError(name)
        else:
            try:
                return self._fields[name].value
            except KeyError:
                raise AttributeError(self._key_error_string(name))

    def __contains__(self, name: str) -> bool:
        return name in self._fields

    def __len__(self) -> int:
        return len(self._fields)

    def __deepcopy__(self, memo):
        clone = type(self)()
        clone._fields = copy.deepcopy(self._fields)
        return clone

    def keys(self) -> List[str]:
        return sorted(self._fields.keys())

    def items(self) -> List[Tuple[str, Any]]:
        """Returns (key, value) pairs sorted by keys."""
        return [(key, getattr(self, key)) for key in self.keys()]

    def set(self, **kwargs) -> 'Config':
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def debug_string(self, *, kv_separator=': ', field_separator='\n'):
        lines = []
        self.visit(lambda key, val: lines.append(f'{key}{kv_separator}{val}'))
        prefix = 'Frozen!\n' if self._frozen else ''
        return prefix + field_separator.join(lines)

    def __str__(self):
        return self.debug_string()

    def __repr__(self):
        return self.debug_string(kv_separator=':', field_separator='; ')

    def visit(self,
              visit_fn: Callable[[str, Any], None],
              enter_fn: Optional[Callable[[str, Any, Optional[List]], Optional[List]]] = None,
              exit_fn: Optional[Callable[[str, Any], None]] = None):
        """Recursively visits objects within this Config instance.

        Visit can traverse Config, lists, tuples, dataclasses, and namedtuples.
        By default, visit_fn is called on any object we don't know how to
        traverse into, like an integer or a string. enter_fn and exit_fn are
        called on objects we can traverse into, like Config, lists, tuples, dicts,
        dataclasses, and namedtuples. We call enter_fn before traversing the object,
        and exit_fn when we are finished.

        A default enter function will be used if enter_fn is None. The default
        function returns None if the value is not a Config, list, tuple, dict,
        dataclass, or namedtuple, otherwise a list of (subkey, subval) pairs to
        traverse.

        Each subkey returned by the default enter function has one of the following
        forms:
          key.subkey when traversing Config objects
          key[1] when traversing lists/tuples/ranges
          key[subkey] when traversing dicts, dataclasses, or namedtuples

        enter_fn, if not None, takes key, value, and the return value of the
        default enter function and returns either None or a list of (subkey, subval)
        pairs to traverse. This allows the user to override the entry decision or
        key format of the default function.

        Args:
          visit_fn: Called on every object for which enter_fn returns None.
          enter_fn: If not None, called on every object. If this function returns
            None, we call visit_fn and do not enter the object.
          exit_fn: Called after an enter-able object has been traversed.
        """
        if not enter_fn:
            enter_fn = lambda key, val, items: items
        if not exit_fn:
            exit_fn = lambda key, val: None

        def _visit(key: str, val: Any):
            val_items = enter_fn(key, val, _default_enter_fn(key, val))
            if val_items is None:
                visit_fn(key, val)
            else:
                for subkey, subval in val_items:
                    _visit(subkey, subval)
                exit_fn(key, val)

        def _default_enter_fn(key: str, val: Any):
            if isinstance(val, Config):
                return [(_SubKey(key, k), v) for k, v in val.items()]
            elif isinstance(val, dict):
                return [(f'{key}[{k}]', v) for k, v in val.items()]
            elif dataclasses.is_dataclass(val):
                return _default_enter_fn(key, val.__dict__)
            elif _is_named_tuple(val):
                return _default_enter_fn(key, val._asdict())
            elif isinstance(val, (list, tuple, range)):
                return [(f'{key}[{i}]', v) for i, v in enumerate(val)]
            else:
                return None  # do not enter 'val'

        def _SubKey(key, subkey):
            if key:
                return f'{key}.{subkey}'
            return subkey

        _visit('', self)

    def _similar_keys(self, name: str) -> List[str]:
        """Return a list of field keys that are similar to name."""

        def _Overlaps(name: str, key: str) -> float:
            """The fraction of 3-char substrings in <name> that appear in key."""
            matches = 0
            trials = 0
            for i in range(len(name) - 3):
                trials += 1
                if name[i:i + 3] in key:
                    matches += 1
            if trials:
                return float(matches) / trials
            return 0

        return [key for key in self.keys() if _Overlaps(name, key) > 0.5]

    def _key_error_string(self, name: str) -> str:
        similar = self._similar_keys(name)
        if similar:
            return f'{name} (did you mean: [{", ".join(sorted(similar))}])'
        return f'{name} (keys are {self.keys()})'


class InstantiableConfig(Config):

    def __init__(self, cls: Optional[type] = None) -> None:
        super().__init__()
        self.define('cls', cls, 'Cls that this Config object is associated with.')

    @staticmethod
    def for_class(cls):
        config = InstantiableConfig(cls)
        init_sig = inspect.signature(cls.__init__)
        for name, param in init_sig.parameters.items():
            if name == 'self':
                continue
            config.define(name, param.default, f'The argument {name} for {cls}.__init__().')
        return config

    def instantiate(self, **kwargs) -> Any:
        """Instantiate an instance that this Config is configured for.

        It's common for classes to have a @classmethod called Config that returns
        a pre-made InstantiableConfig, like this:

          class MyObject:

            @classmethod
            def default_config(cls):
              cfg = config.InstantiableConfig(cls=cls)
              cfg.define('weight', 0.2, 'Training weight.')
              return cfg

          # Create a MyObject with config.
          cfg = MyObject.default_config().set(weight=0.9)
          obj = cfg.Instantiate()

        By convention, anything that configures the behavior of your class
        should be stored in this Config object. However, your class may also use
        shared state objects which aren't really part of the config, like a shared lock.
        These can be passed as extra arguments to Instantiate.

        Example:
          lock = threading.Lock()
          config = MyObject.default_config()
          obj_a = config.Instantiate(lock=lock)
          obj_b = config.Instantiate(lock=lock)

        Args:
          **kwargs: Additional keyword arguments to pass to the constructor in
            addition to this Config object.

        Returns:
          A constructed object where type(object) == cls.
        """
        assert self.cls is not None

        if issubclass(self.cls, Configurable):
            # The class initializer is expected to support initialization using Config.
            return self.cls(self, **kwargs)
        else:
            # Initialize by passing config fields as kwargs.
            return self.cls(**{k: v for k, v in self.items() if k != 'cls'}, **kwargs)


class Configurable:

    @classmethod
    def default_config(cls):
        cfg = InstantiableConfig(cls)
        return cfg

    def __init__(self, config: Config):
        self.__dict__['_config'] = copy.deepcopy(config).freeze()

    @property
    def config(self):
        return copy.deepcopy(self._config)

    def __repr__(self):
        return repr(self._config)

import contextlib
import dataclasses
import functools
import numbers
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import jax
import numpy
from flax import serialization
from jax import numpy as jnp
from jax.experimental import pjit
from jax.tree_util import register_pytree_node_class

Tensor = jnp.ndarray
# Recursive type annotations not supported by pytype yet.
NestedTree = Union[Any, Dict[str, Any]]  # Union[Any, Dict[str, "NestedTree"]]
NestedTensor = Union[Tensor, Dict[str, Any]]  # Union[Tensor, Dict[str, "NestedTensor"]]


_enable_numeric_checks = False


@contextlib.contextmanager
def numeric_checks(enabled: bool = True):
    old_state = _enable_numeric_checks

    def switch(value):
        global _enable_numeric_checks
        _enable_numeric_checks = value
        jax.config.update("jax_debug_nans", value)

    switch(enabled)
    yield
    switch(old_state)


def check_numerics(x: Tensor, msg_fmt: str = "", **msg_kwargs):
    """Checks that all elements in `x` are finite."""
    global _enable_numeric_checks
    if _enable_numeric_checks:
        assert bool(jnp.isfinite(x).all()), f"Check numerics {msg_fmt.format(**msg_kwargs)}: {x}"
    return x


def shapes(nested_tensor: NestedTensor) -> NestedTree:
    """Returns a tree of the same structure as `nested_tensor` but with corresponding shapes instead of tensors."""
    return jax.tree_map(lambda x: getattr(x, "shape", x), nested_tensor)


def tree_paths(tree: NestedTree, separator="/") -> NestedTree:
    """Returns a tree of the same structure as `nested_tensor` but with corresponding paths instead of values.

    E.g.,
        tree_paths({'a': 1, 'b': [2, {'c': 3}]}) = {'a': 'a', 'b': ['b/0', {'c': 'b/1/c'}]}

    Args:
        tree: a nested structure.
        separator: the separator between parts of a path.

    Returns:
        A nested structure with the same structure as `tree`.
    """

    def _concat(prefix, suffix):
        return f"{prefix}{separator}{suffix}" if prefix else f"{suffix}"

    def visit(tree, prefix):
        if isinstance(tree, dict):
            return type(tree)((k, visit(v, _concat(prefix, k))) for k, v in tree.items())
        elif is_named_tuple(tree):
            return type(tree)(**visit(tree._asdict(), prefix))
        elif isinstance(tree, (list, tuple)):
            return type(tree)([visit(v, _concat(prefix, k)) for k, v in enumerate(tree)])
        else:
            return prefix

    return visit(tree, "")


@dataclasses.dataclass
class PathAndValue:
    path: str
    value: Any


def flatten_items(tree: NestedTensor, separator="/") -> Sequence[Tuple[str, Tensor]]:
    """Flattens `tree` and returns a list of (path, value) pairs."""
    paths = tree_paths(tree, separator=separator)
    paths_and_values = jax.tree_map(lambda path, value: PathAndValue(path, value), paths, tree)
    flat_paths_and_values, _ = jax.tree_flatten(paths_and_values)
    return list((pv.path, pv.value) for pv in flat_paths_and_values)


@register_pytree_node_class
class VDict(dict):
    """A dict with Tensor leaf nodes whose values should be vectorized."""

    def __repr__(self):
        return "VDict(%r)" % super().__repr__()

    def tree_flatten(self):
        return (self.values(), self.keys())

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values))


# Register VDict as a dict for Flax serialization.
serialization.register_serialization_state(
    VDict,
    ty_to_state_dict=serialization._dict_state_dict,
    ty_from_state_dict=serialization._restore_dict,
)


def vectorized_tree_map(fn, tree, *rest):
    """Similar to jax.tree_map(), but vectorizes `fn` on VDict's."""

    def vectorized_fn(*nodes):
        if type(nodes[0]) == VDict:
            nodes = [dict(**node) for node in nodes]
            result = jax.vmap(functools.partial(vectorized_tree_map, fn))(*nodes)
            return VDict(**result)
        return fn(*nodes)

    return jax.tree_map(vectorized_fn, tree, *rest, is_leaf=lambda t: isinstance(t, VDict))


def is_named_tuple(x):
    """Returns whether an object is an instance of a collections.namedtuple.

    Examples::
      is_named_tuple((42, 'hi')) ==> False
      Foo = collections.namedtuple('Foo', ['a', 'b'])
      is_named_tuple(Foo(a=42, b='hi')) ==> True

    Args:
      x: The object to check.
    """
    return isinstance(x, tuple) and hasattr(x, "_fields") and hasattr(x, "_asdict")


def as_tensor(x):
    """Converts `x` to Tensor recursively.

    Args:
        x: a jnp array, numpy array, TF/PyTorch Tensor, or a nested structure of arrays or Tensors.

    Returns:
        A nested structure with the same structure as `x` but with values converted to Tensors.
    """
    if isinstance(x, Tensor):
        return x
    if isinstance(x, (numbers.Number, numpy.ndarray)):
        return jnp.asarray(x)
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "numpy"):
        return jnp.asarray(x.numpy())
    if isinstance(x, (Mapping, Sequence)):
        return jax.tree_map(as_tensor, x)
    raise NotImplementedError(f"{type(x)}: {x}")


def with_sharding_constraint(x, axis_resources):
    mesh = jax.experimental.maps.thread_resources.env.physical_mesh
    if mesh.empty:
        return x
    return pjit.with_sharding_constraint(x, axis_resources)

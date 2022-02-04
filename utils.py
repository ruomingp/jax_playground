import numbers
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import jax
import numpy
from jax import numpy as jnp

Tensor = jnp.ndarray
# Recursive type annotations not supported by pytype yet.
NestedTree = Union[Any, Dict[str, Any]]  # Union[Any, Dict[str, "NestedTree"]]
NestedTensor = Union[Tensor, Dict[str, Any]]  # Union[Tensor, Dict[str, "NestedTensor"]]

enable_numeric_checks = False


def check_numerics(x: Tensor, msg_fmt: str = "", **msg_kwargs):
    """Checks that all elements in `x` are finite."""
    global enable_numeric_checks
    if enable_numeric_checks:
        assert bool(jnp.isfinite(x).all()), f"Check numerics {msg_fmt.format(**msg_kwargs)}: {x}"
    return x


def shapes(nested_tensor: NestedTensor) -> NestedTree:
    """Returns a tree of the same structure as `nested_tensor` but with corresponding shapes instead of tensors."""
    return jax.tree_map(lambda x: x.shape, nested_tensor)


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
            return {k: visit(v, _concat(prefix, k)) for k, v in tree.items()}
        elif is_named_tuple(tree):
            return type(tree)(**visit(tree._asdict(), prefix))
        elif isinstance(tree, (list, tuple)):
            return type(tree)([visit(v, _concat(prefix, k)) for k, v in enumerate(tree)])
        else:
            return prefix

    return visit(tree, "")


def flatten_items(tree: NestedTensor, separator="/") -> Sequence[Tuple[str, Tensor]]:
    """Flattens `tree` and returns a list of (path, value) pairs."""
    paths = tree_paths(tree, separator=separator)
    flat_paths, _ = jax.tree_flatten(paths)
    flat_values, _ = jax.tree_flatten(tree)
    return list(zip(flat_paths, flat_values))


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

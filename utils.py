import jax
from jax import numpy as jnp
from typing import Dict, Mapping, Sequence, TypeVar, Union


Tensor = jnp.ndarray
NestedTensor = Dict[str, Union[Tensor, "NestedTensor"]]


def tree_paths(tree, separator="/"):
    def _concat(prefix, suffix):
        return f"{prefix}{separator}{suffix}" if suffix else f"{prefix}"

    if isinstance(tree, Mapping):
        return {
            k: jax.tree_map(lambda value: _concat(k, value), tree_paths(v))
            for k, v in tree.items()
        }
    elif is_named_tuple(tree):
        return tree_paths(tree._asdict())
    elif isinstance(tree, Sequence):
        return [
            jax.tree_map(lambda value: _concat(k, value), tree_paths(v))
            for k, v in enumerate(tree)
        ]
    else:
        return ""


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

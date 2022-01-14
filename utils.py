import jax
from jax import numpy as jnp
from typing import Dict, Mapping, Sequence, TypeVar, Union


Tensor = jnp.ndarray
NestedTensor = Dict[str, Union[Tensor, "NestedTensor"]]

enable_numeric_checks = False


def check_numerics(x: Tensor, msg_fmt: str = "", **msg_kwargs):
    global enable_numeric_checks
    if enable_numeric_checks:
        assert bool(
            jnp.isfinite(x).all()
        ), f"Check numerics {msg_fmt.format(**msg_kwargs)}: {x}"
    return x


def shapes(nested_tensor: NestedTensor) -> NestedTensor:
    return jax.tree_map(lambda x: x.shape, nested_tensor)


def tree_paths(tree, separator="/"):
    def _concat(prefix, suffix):
        return f"{prefix}{separator}{suffix}" if prefix else f"{suffix}"

    def visit(tree, prefix):
        if isinstance(tree, Mapping):
            return {
                k: visit(v, _concat(prefix, k)) for k, v in tree.items()
            }
        elif is_named_tuple(tree):
            return visit(tree._asdict(), prefix)
        elif isinstance(tree, Sequence):
            return [visit(v, _concat(prefix, k)) for k, v in enumerate(tree)]
        else:
            return prefix

    return visit(tree, "")


def flatten_items(tree):
    paths = tree_paths(tree)
    return zip(jax.tree_flatten(paths), jax.tree_flatten(tree))


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

import collections
import dataclasses
import typing

import jax
from jax import numpy as jnp

input_stack = []
output_queue = []


def f(x):
    return x + 0.1


def f_from_stack():
    print("f_from_stack")
    x = input_stack.pop(-1)
    return f(x)


def f_via_stack(x):
    input_stack.append(x)
    y = f_from_stack()
    output_queue.append(y)
    return y


# print(jax.jit(f)(jnp.ones((2, 4), dtype=jnp.float32)))
print(jax.jit(f_via_stack)(jnp.ones((2, 4), dtype=jnp.float32)))
print(f"output={output_queue}")
print(jax.jit(f_via_stack)(jnp.zeros((2, 4), dtype=jnp.float32)))
print(f"output={output_queue}")
print(jax.jit(f_via_stack)(jnp.zeros((2, 3), dtype=jnp.float32)))
print(f"output={output_queue}")


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Point:
    x: int
    y: int

    def tree_flatten(self):
        return ((self.x, self.y), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


print(jax.tree_map(lambda z: z + 1, Point(x=1, y=2)))

Point = collections.namedtuple("Point", ("x", "y"))
print(jax.tree_map(lambda z: z + 1, Point(x=1, y=2)))

Point = typing.NamedTuple("Point", (("x", int), ("y", int)))
print(jax.tree_map(lambda z: z + 1, Point(x=1, y=2)))


class Point(typing.NamedTuple):
    x: int
    y: int


print(jax.tree_map(lambda z: z + 1, Point(x=1, y=2)))

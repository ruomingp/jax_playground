"""Misc tests about Jax."""

import collections
import dataclasses
import typing

import jax
import jaxlib
import numpy as np
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


prng_key1 = jax.random.PRNGKey(123)
keys1 = []
for _ in range(3):
    prng_key1, key = jax.random.split(prng_key1)
    keys1.append(key)

keys2 = jax.random.split(jax.random.PRNGKey(123), 4)
print(keys1)
print(keys2)


x = jax.lax.broadcasted_iota("int32", [4, 2], 0)
print(x)
x = jnp.roll(x, 1, axis=0)
print(x)

x = jnp.zeros([3], dtype=jnp.float32)
print(x.nbytes)
print(jnp.floor(x).dtype)


x = jax.random.normal(jax.random.PRNGKey(123), [10, 6, 4], dtype=jnp.bfloat16)


def center(x, unused):
    x_dtype = x.dtype
    x = x.astype(jnp.float32)
    x_mean = x.mean(axis=-1, keepdims=True)
    x -= x_mean
    return x.astype(x_dtype), x_mean


center1 = center(x, 0)
center2 = jax.lax.scan(center, init=x, xs=jnp.arange(1))
np.testing.assert_array_equal(center1[0], center2[0])
np.testing.assert_array_equal(center1[1], center2[1][0])

print(type(x))
print(dir(x))
print(x.unsafe_buffer_pointer())


class MyArray(jaxlib.xla_extension.DeviceArray):
    def __init__(self, x: jnp.ndarray, axis_names):
        super.__init__(x)
        self.axis_names = axis_names


y = MyArray(x, ("b", "l", "d"))
print(y.unsafe_buffer_pointer())

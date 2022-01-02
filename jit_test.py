import jax
from jax import numpy as jnp

input_stack = []
output_queue = []


def f(x):
    return x + 0.1


def f_from_stack():
    print('f_from_stack')
    x = input_stack.pop(-1)
    return f(x)


def f_via_stack(x):
    input_stack.append(x)
    y = f_from_stack()
    output_queue.append(y)
    return y


# print(jax.jit(f)(jnp.ones((2, 4), dtype=jnp.float32)))
print(jax.jit(f_via_stack)(jnp.ones((2, 4), dtype=jnp.float32)))
print(f'output={output_queue}')
print(jax.jit(f_via_stack)(jnp.zeros((2, 4), dtype=jnp.float32)))
print(f'output={output_queue}')
print(jax.jit(f_via_stack)(jnp.zeros((2, 3), dtype=jnp.float32)))
print(f'output={output_queue}')

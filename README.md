# The Ajax Library for Deep Learning

Ajax is a library for deep learning built upon Jax and GSPMD intended to
support large-scale training of foundation models.

## Principles

* Simplicity: explicitness over magic
* Flexibility: provide building blocks instead of frameworks

## Design Choices

### Configuration

(For people familiar with Lingvo's configuration system, Ajax's config library is
very similar except for the terminology change of "parameter" to "config".)



### GSPMD

We choose to represent all computation via global sharding instead of supporting
multiple modes (data parallel vs. global sharding).

Specifically, all tensors and computation expressed in Jax describes
the global tensors and computation,
which are sharded on a device mesh along the data and model axis.
To represent a pure data-parallel computation,
one just sets the model dims of the device mesh to 1.

TODO(ruoming): add an example here.

A notable exception is input processing, which is not expressed in Jax and describes not the global
computation but what happens on each host.

### Invocation Context

One of Jax's appeals is the functional programming paradigm---users
are encouraged to write pure functions without side inputs and outputs.

In practice, we found that certain inputs and outputs need to propagated forth and back
between any pair of parent and child layers.
These include model parameters,
whether the computation is in the training or evaluation mode (`is_training`),
and pseudo-random generator key as parts of layer inputs,
and summaries and auxiliary parameter updates (e.g., moving averages) as parts of layer outputs.
Written explicitly, the code will look like

```python
def forward(self, is_training, prng_key, parameters, x):
    ...
    # For each child...
    prng_key, child_key = jax.random.split(prng_key)
    summaries = {}
    parameter_updates = {}
    x, child_aux_outputs = self.child.forward(
        is_training, child_key, parameters["child"], x)
    summaries["child"] = child_aux_outputs["summaries"]
    parameter_updates["child"] = child_aux_outputs["parameter_updates"]
    ...
    return x, dict(summaries=summaries, parameter_updates=parameter_updates)
```

This seems unnecessarily verbose.
So we took a page from Flax and Lingvo's approach
and use a thread-local stack of InvocationContext to represent the implicit inputs and outputs.
This allows us to write code in a concise style as in PyTorch/Flax:

```python
def forward(self, x):
    ...
    # Implicit inputs can be accessed via self.{is_training, prng_key, parameters}.
    x = self.child(x)
    # Implicit outputs can be added via self.add_{summary, parameter_update}().
    
    # User can override implicit inputs, e.g., to invoke the teacher module in the evaluation mode.
    with set_current_context(current_context().clone(is_training=False)):
        y = self.teacher(x)
    ...
    return x
```

One exception is at the root level,
where we need pure functions for jit, pjit, and differentiation.
We provide a `functional()` method,
which converts any module method invocation into a functional API
with explicit inputs and outputs.
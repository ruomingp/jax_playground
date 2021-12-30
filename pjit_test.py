import jax
import jax.numpy as jnp
from jax.experimental import maps
from jax.experimental import mesh_utils
from jax.experimental import PartitionSpec
from jax.experimental.pjit import pjit
import numpy as np

print(jax.__version__)
print(jax.devices())

mesh_shape = (4, 2)
devices = mesh_utils.create_device_mesh(mesh_shape)
mesh = maps.Mesh(devices, ('data', 'model'))
print(mesh)

x = np.arange(8 * 2).reshape(8, 2)
print(x)

rng_key = jax.random.PRNGKey(42)
params = {}
param_spec = {}
for k in ('w1', 'w2'):
    rng_key, sub_key = jax.random.split(rng_key)
    params[k]= jax.random.normal(sub_key, shape=(2, 6))
    param_spec[k] = PartitionSpec(None, 'model')

x_spec = PartitionSpec('data', None)
y_spec = PartitionSpec('data', None)
# y_spec = PartitionSpec('data', 'model')

def mlp(params, x):
    x = x @ params['w1']
    x = jnp.fmax(x, 0)
    x = x @ jnp.transpose(params['w2'])
    return x


# f = pjit(mlp, in_axis_resources=(param_spec, x_spec), out_axis_resources=y_spec, donate_argnums=(0,))
f = pjit(mlp, in_axis_resources=(param_spec, x_spec), out_axis_resources=y_spec, donate_argnums=(0,))
 
# Sends data to accelerators based on partition_spec
with maps.mesh(mesh.devices, mesh.axis_names):
      y = f(params, x)
print(y)

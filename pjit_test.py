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

input_data = np.arange(8 * 2).reshape(8, 2)
print(input_data)

in_spec = PartitionSpec('data', None)
out_spec = PartitionSpec('data', 'model')

f = pjit(lambda x: x, in_axis_resources=in_spec, out_axis_resources=out_spec)
 
# Sends data to accelerators based on partition_spec
with maps.mesh(mesh.devices, mesh.axis_names):
      data = f(input_data)
print(data)

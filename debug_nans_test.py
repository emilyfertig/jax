import functools
import jax
import jax.numpy as jnp

@jax.jit
def g(z, w):
  a = z + w
  b = z - w
  return a / b

# @functools.partial(jax.jit, inline=True)
@jax.jit
def f(x, y):
  a = x * y
  # b = g(x, y)
  b = (x + y) / (x - y)
  c = a + 2
  return a + b * c

x = jnp.array([2., 0.])
y = jnp.array([3., 0])

with jax.debug_nans(True):
# with jax.debug_nans(False):
  print(f(x, y))
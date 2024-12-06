import jax
import jax.numpy as jnp

# (1) NaNs on the forward pass
@jax.jit
def f(x):
  y = jnp.square(x)
  return jnp.log(-y)

# (2) NaNs on the forward pass but nested
@jax.jit
def g(x):
  return f(x - 2.)

x = jnp.array([2., 0.])
z = jnp.zeros_like(x)

# (3) NaNs on the backward pass
out, f_vjp = jax.vjp(f, x)
# (4) ...and nested.
out, g_vjp = jax.vjp(g, x)

# (5) NaNs in forward autodiff
f_jvp = lambda: jax.jvp(f, [z], [jnp.ones_like(x)])

with jax.debug_nans(True):
  # jax.print_environment_info()
  f(x)
  # g(x)
  # f_vjp(x)
  # print(g_vjp(x))
  # f_jvp()

  # Handle literal case!
  # jax.jit(lambda: jnp.nan)()

  # jax.jit(f_vjp)(z)
  # jax.jit(f_jvp)()
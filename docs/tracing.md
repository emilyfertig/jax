---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(tracing)=
# Tracing

<!--* freshness: { reviewed: '2024-12-01' } *-->

In the process of transforming functions, JAX replaces some function arguments with special tracer values. {ref}`jit-how-jax-transformations-work` gave a short introduction to tracing in the context of JIT

Recall what happens if you ``print`` a JAX array inside of a JIT-compiled function:

```{code-cell}
import jax
import jax.numpy as jnp

def func(x):
  print(x)
  return jnp.cos(x)

res = jax.jit(func)(0.)
print(res)
```

The above code does return the correct value ``1.`` but it also prints ``Traced<ShapedArray(float32[])>`` for the value of ``x``. Normally, JAX handles these tracer values internally in a transparent way, e.g., in the numeric JAX primitives that are used to implement the ``jax.numpy`` functions. This is why ``jnp.cos`` works in the example above.

More precisely, a **tracer** value is introduced for the argument of a JAX-transformed function, except the arguments identified by special parameters such as ``static_argnums`` for `jax.jit` or ``static_broadcasted_argnums`` for `jax.pmap`. Typically, computations that involve at least a tracer value will produce a tracer value. Besides tracer values, there are **regular** Python values: values that are computed outside JAX transformations, or arise from above-mentioned static arguments of certain JAX transformations, or computed solely from other regular Python values.  These are the values that are used everywhere in absence of JAX transformations.

A tracer value carries an **abstract** value, e.g., ``ShapedArray`` with information about the shape and dtype of an array. We will refer here to such tracers as **abstract tracers**. Some tracers, e.g., those that are introduced for arguments of autodiff transformations, carry abstract values that actually include the regular array data, and are used, e.g., for resolving conditionals. We will refer here to such tracers as **concrete tracers**. Tracer values computed from these concrete tracers, perhaps in combination with regular values, result in concrete tracers.  A **concrete value** is either a regular value or a concrete tracer.

Most often values computed from tracer values are themselves tracer values.  There are very few exceptions, when a computation can be entirely done using the abstract value carried by a tracer, in which case the result can be a regular value. For example, getting the shape of a tracer with ``ShapedArray`` abstract value. Another example is when explicitly casting a concrete tracer value to a regular type, e.g., ``int(x)`` or ``x.astype(float)``.  Another such situation is for ``bool(x)``, which produces a Python bool when concreteness makes it possible. That case is especially salient because of how often it arises in control flow.

Here is how the transformations introduce abstract or concrete tracers:

* `jax.jit`: introduces **abstract tracers** for all positional arguments except those denoted by ``static_argnums``, which remain regular values.
* `jax.pmap`: introduces **abstract tracers** for all positional arguments except those denoted by ``static_broadcasted_argnums``.
* `jax.vmap`, `jax.make_jaxpr`, `xla_computation`: introduce **abstract tracers** for all positional arguments.
* `jax.jvp` and `jax.grad` introduce **concrete tracers** for all positional arguments. An exception is when these transformations are within an outer transformation and the actual arguments are themselves abstract tracers; in that case, the tracers introduced by the autodiff transformations are also abstract tracers.
* All higher-order control-flow primitives (`lax.cond`, `lax.while_loop`, `lax.fori_loop`, `lax.scan`) when they process the functionals introduce **abstract tracers**, whether or not there is a JAX transformation in progress.

All of this is relevant when you have code that can operate only on regular Python values, such as code that has conditional control-flow based on data:

```{code-cell}
def divide(x, y):
  return x / y if y >= 1. else 0.
```

If we want to apply `jax.jit`, we must ensure to specify ``static_argnums=1`` to ensure ``y`` stays a regular value. This is due to the boolean expression ``y >= 1.``, which requires concrete values (regular or tracers). The same would happen if we write explicitly ``bool(y >= 1.)``, or ``int(y)``, or ``float(y)``.

Interestingly, ``jax.grad(divide)(3., 2.)``, works because `jax.grad` uses concrete tracers, and resolves the conditional using the concrete value of ``y``.

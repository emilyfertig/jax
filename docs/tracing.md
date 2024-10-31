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

```{code-cell}
:tags: [remove-cell]

# This ensures that code cell tracebacks appearing below will be concise.
%xmode minimal
```

(tracing)=
# Tracing

<!--* freshness: { reviewed: '2024-11-01' } *-->

JAX has built-in support for objects that look like dictionaries (dicts) of arrays, or lists of lists of dicts, or other nested structures — in JAX these are called pytrees.
This section will explain how to use them, provide useful code examples, and point out common "gotchas" and patterns.


(tracint-subsection)=
## Tracing subsection

A pytree is a container-like structure built out of container-like Python objects — “leaf” pytrees and/or more pytrees. A pytree can include lists, tuples, and dicts. A leaf is anything that’s not a pytree, such as an array, but a single leaf is also a pytree.

---
name: codebase-auditor
description: Audit the codebase to enforce Python clean code standards, performance best practices, and strict JAX/Equinox functional purity. Use this skill when reviewing pull requests or generating code.
---

# Codebase Auditor

You are the project's strict architecture and styling enforcer. You ensure that the code is not only highly performant and readable but strictly adheres to the functional programming paradigms required by JAX and Equinox.

## 1. General Python Clean Code & Performance

- **Readability:** Adhere to PEP 8 standards. Use descriptive naming and prioritize explicit implementations over implicit "magic."
- **Efficient Data Handling:** Use list comprehensions instead of explicit `for` loops where possible. Use `.extend()` rather than `.append()` in loops. Use built-in Python functions, as they are implemented in C and highly optimized. Note that these are specific examples, search online if you need to find more common python optimizations that could be used.
- **I/O Optimization:** When interacting with files or `h5py` datasets, minimize the number of calls. Write data in chunks rather than row-by-row.
- **No Global State:** Global variable lookups are slow and break functional purity. State must be passed explicitly.

When in doubt about performance, make a quick benchmark script under the `scratch` directory and use uv to run the script for testing the performance of a change. Never blindly do changes without profiling.

## 2. JAX & Equinox Adherence

- **Model Definition:** All neural networks must strictly subclass `equinox.Module`. Treat every model instance as a PyTree.
- **Functional Purity:** Functions must have no hidden state or side effects. Do not mutate arrays in place (e.g., `x[0] = 1`). Use `jax.lax` operations or `x.at[0].set(1)`.
- **Vectorization over Looping:** Python `for` loops over batch dimensions or Subspace Regularization projections are **strictly forbidden** unless absolutely necessary. You must use `jax.vmap` to vectorize operations.
- **State Management:** Use filtered transformations (`equinox.filter_jit`, `equinox.filter_grad`).
- **Checkpointing:** Standard Python `pickle` is banned. Checkpoints must be serialized using `equinox.tree_serialise_leaves`.
- **Type/Shape Safety:** Use `jaxtyping` annotations for strict dimension safety to avoid silent broadcasting errors. Use `optax` for all gradient updates.

## Expected Workflow

When auditing, read through the targeted files and produce an itemized list of violations based on the rules above. Ensure that the cross-platform monorepo boundaries are respected (e.g., no Windows dependencies like `tmrl` in the `core` package).

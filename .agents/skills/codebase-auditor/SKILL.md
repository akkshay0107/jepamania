---
name: codebase-auditor
description: Audit the codebase to enforce Python clean code standards, performance best practices, and strict JAX/Equinox functional purity. Use this skill when reviewing pull requests or generating code.
---

# Codebase Auditor

Ensure that the code is not only highly performant and readable but strictly adheres to the functional programming paradigms required by JAX and Equinox.
NOTE: Do not apply this to any library source code that exists in the virtual environment or the scratch directory. Editing code in virtual environments is strictly off limits.

## 1. Python Clean Code & Performance

- **Readability:** Follow PEP 8. Use explicit implementations and descriptive naming.
- **Data & I/O:** Use optimized built-ins/list comprehensions where possible. Batch/chunk file and `h5py` operations.
- **Purity:** Avoid global state.
- **Profiling:** When testing performance changes, create and run a benchmark script in the `scratch` directory using `uv`. Do not make optimizations without profiling.

## 2. JAX & Equinox Adherence

- **Model Definition:** All neural networks must strictly subclass `equinox.Module`. Treat every model instance as a PyTree.
- **Functional Purity:** Functions must have no hidden state or side effects. Do not mutate arrays in place (e.g., `x[0] = 1`). Use `jax.lax` operations or `x.at[0].set(1)`.
- **Vectorization over Looping:** Python `for` loops over batch dimensions or Subspace Regularization projections are **strictly forbidden** unless absolutely necessary. You must use `jax.vmap` to vectorize operations.
- **State Management:** Use filtered transformations (`equinox.filter_jit`, `equinox.filter_grad`).
- **Checkpointing:** Standard Python `pickle` is banned. Checkpoints must be serialized using `equinox.tree_serialise_leaves`.
- **Type/Shape Safety:** Use `jaxtyping` annotations for strict dimension safety to avoid silent broadcasting errors. Use `optax` for all gradient updates.

## 3. Static LSP Code Checks and Cleanup

- **Linter (Ruff):**
  - Run lint check: `uv run ruff check .`
  - Auto-fix lint violations: `uv run ruff check --fix .`
  - Format code files: `uv run ruff format .`
- **Type Checking (Pyright):** Use pyright to verify type annotations.
  - Run type checking: `uv run pyright .`
  - Go over all the errors and warnings that it provides. If they are meaningful and can be fixed easily, make the necessary edits to fix them. If they are harmless and tedious to fix, prefer using comments to ignore them and note them down in your response in case the warning is relevant in the future.

## 4. Commenting Style Guideline

- Ensure that the comments are meaningful and do not explain the obvious. Comments should provide additional information or clarification that is not directly observable from the code. Prefer explicit names to avoid using comments that explain ambiguous naming conventions.
- Avoid using numbered lists or section headers. The code should be structured in a way that they are not necessary.
- Use comments to explain why a certain block of code exists or is used rather than what the code does. In places where the code is complex, and the way the code is written obscures its actual functionality, it is ok to explain what the code does.
- The above are not hard rules that need to be adhered to, but general guidelines. Feel free to go against them if needed.

## Expected Workflow

Audit target files, generate an itemized list of violations, and respect cross-platform boundaries (e.g., no Windows/`tmrl` dependencies in `core`).

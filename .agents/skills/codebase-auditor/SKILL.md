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
  - Go over all the errors and warnings that it provides. If they point to real logic or safety bugs, make the necessary code edits to fix them. For simple or harmless type checking issues (e.g., union return types from external libraries like `h5py` or minor attribute/indexing mismatches), prefer using `# pyright: ignore[...]` or `# type: ignore` comments instead of verbose runtime `typing.cast(...)` wrapper calls. Note ignored issues down in your response in case the warning is relevant in the future.

## 4. Commenting Style Guideline

These rules are mandatory requirements of the codebase auditing process. Every audited file must be scanned specifically for comment-style violations.

- **Do not explain the obvious:** Comments must provide non-trivial information or clarifications that are not directly observable from reading the code itself.
- **Prefer self-documenting code:** Use descriptive variable/function/class names rather than comments explaining ambiguous naming conventions.
- **Explain the "Why," not the "What":** Comments must explain *why* a block of code exists or why a specific approach was taken, rather than *what* the code does. Writing comments detailing *what* the code does is only permitted if the implementation is highly complex/obscured.
- **Structure code cleanly:** Avoid using numbered lists or markdown-style section headers inside comment blocks. The code structure must be self-explanatory without these markers.
- **Module Docstrings for Large Files:** Each large file (> 300 lines of code) that handles multiple functionalities must have a top-level module docstring briefly explaining the objective of the file and what it handles.
- **Concise Docstrings for Small Functions:** Small functions should be named in a self-descriptive manner. If additional context is needed, include a brief one-line docstring.
- **Structured Docstrings for Large Functions:** Every large function (> 50 lines of code) that accepts multiple parameters must include a structured docstring in the following format:

```python
"""<Brief Description of function>

Arguments:
  <one line brief for each argument>

Returns:
  <description of what the function returns>
"""
```

## Expected Workflow

1. Identify target files for audit.
2. Perform lint/type checks as specified in Section 3.
3. Scan the code specifically for comment-style compliance against the strict rules in Section 4.
4. Generate a comprehensive itemized list of JAX purity, Python performance, type safety, and commenting style violations.
5. Fix violations or report them explicitly to the user. Respect cross-platform boundaries (e.g., no Windows/`tmrl` dependencies in `core`).

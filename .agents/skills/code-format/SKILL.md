---
name: code-format
description: Enforce Python code standards, formatting conventions, and strict JAX/Equinox conventions and safety directly on the codebase. Use this skill when the user asks to clean up, standardize, or fix style/formatting in code, or after code has been written and needs to be brought into compliance.
---

# Code Formatter

NOTE: Do not apply this to any library source code that exists in the virtual environment or the scratch directory. Editing code in virtual environments is strictly off limits.

## 1. Python Code Style Guidelines

- **Readability:** Follow PEP 8. Use explicit implementations and descriptive naming. Use optimized built-ins or standard library functions if they exist instead of reimplementing features.
- **Logical Spacing:** Use newlines to separate groups of logically distinct statements within a function or block. Don't let unrelated operations run together without visual separation.
- **Dead Code Removal:** Detect and remove unused imports, unused variables/assignments, and unreachable code paths. Do not leave commented-out code blocks in place — remove them.
- **PEP 8 Practices Beyond Formatters:** Ruff and other autoformatters fix whitespace, line length, and import ordering, but they do not enforce the following — check for these manually:
  - **Mutable default arguments:** Never use `def f(x, cache={})` or `def f(x, items=[])`. Default to `None` and initialize inside the function body.
  - **Explicit `is` for singleton comparisons:** Use `is None` / `is not None`, `is True` / `is False` rather than `==`. Never write `if x == None:`.
  - **No bare `except:`:** Always catch a specific exception type. If genuinely catching anything, use `except Exception:` and state why in a comment.
  - **No shadowing builtins:** Don't name variables `list`, `type`, `id`, `input`, `dict`, etc.
  - **Truthiness over explicit comparison:** Prefer `if items:` over `if len(items) > 0:`, and `if not items:` over `if len(items) == 0:`.
  - **Comparisons chained correctly:** Prefer `a < b < c` over `a < b and b < c`.
- **Optional/None Handling:** Avoid threading `Optional`/`None` values deep into JAX-transformed code unless a third-party API genuinely requires it. Validate and resolve `None` at the boundary (e.g. in `__init__` or right after an argument enters a function) and raise an explicit error there, rather than letting `None` propagate into `jit`/`grad`/`vmap`-wrapped logic where its absence causes a confusing trace-time failure instead of a clear one.

## 2. JAX & Equinox Guidelines

- **Model Definition:** All model components must cleanly subclass `equinox.Module`. Declare fields as class-level annotations; mark any non-array/non-trainable field (config values, ints, flags) with `eqx.field(static=True)` so it isn't treated as a pytree leaf.
- **Functional Purity:** Code inside anything wrapped in `jax.jit`, `jax.grad`, `jax.vmap`, or their `eqx.filter_*` equivalents must be pure — no in-place mutation of arrays, no reliance on global/mutable state, no Python-level side effects (printing, logging, appending to external lists) that depend on traced values.
- **Control Flow:** Avoid Python `if`/`for`/`while` whose condition depends on a traced array value inside jitted code. Use `jax.lax.cond` for branching and `jax.lax.scan` or `jax.lax.fori_loop` for loops over traced state instead of Python loops.
- **Transformations:** Prefer `eqx.filter_jit`, `eqx.filter_grad`, and `eqx.filter_vmap` over the raw `jax.*` equivalents when a pytree contains non-array leaves (e.g. `eqx.Module` with static fields, strings, or config objects), since raw `jax.jit` will error or needlessly retrace on those.
- **PRNG Key Management:** Never reuse a `jax.random.PRNGKey`/`jax.random.key`. Split explicitly with `jax.random.split` and thread keys through function arguments rather than storing them as mutable state. Each stochastic operation should receive its own subkey.
- **Device & Dtype Management:** Ensure explicit dtype handling (`jnp.asarray(x, dtype=...)`); do not rely on implicit x64 promotion. Do not hardcode device placement — use `jax.device_put` with an explicit device argument only when placement genuinely needs to be pinned.
- **Checkpointing & Serialization:** Do not use raw `pickle` for model state. Use `eqx.tree_serialise_leaves` / `eqx.tree_deserialise_leaves` for saving and loading Equinox pytrees, and pair with atomic write practices (temporary file plus `os.replace`).
- **Shape & Dtype Safety:** Annotate array-valued fields and function signatures with `jaxtyping` types (e.g. `Float[Array, "batch dim"]`, `Int[Array, "n"]`) rather than bare `Array` or `jnp.ndarray`, so shape and dtype mismatches surface as type errors instead of silent broadcasting bugs. Use `jaxtyping.jaxtyped` combined with `beartype` (or `typeguard`) as a decorator on functions with non-trivial shape contracts if runtime enforcement is available in the environment; otherwise rely on the annotations for static/documentation value only and note this in the summary.
- **Runtime Validation:** Use `__check_init__` on `eqx.Module` subclasses to validate structural invariants (shape relationships, valid ranges, required field combinations) at construction time rather than deep inside methods. Use `eqx.error_if` for runtime checks on traced values inside jitted code, since a plain Python `assert` or `raise` on a traced value will not behave correctly under `jit`.

## 3. Static LSP Code Checks and Cleanup

- **Linter (Ruff):**
  - Run lint check: `uv run ruff check <affect directory>`
  - Auto-fix lint violations: `uv run ruff check --fix <affected directory>`
  - Format code files: `uv run ruff format <affected directory>`
- **Type Checking (Pyright):** Use pyright to verify type annotations.
  - Run type checking: `uv run pyright <affected directory>`
  - Go over all the errors and warnings that it provides. If they point to real logic or safety bugs, make the necessary code edits to fix them.

To handle type errors that are not logical bugs, use `# type: ignore` comments for one-off errors. If an ambiguous variable causes multiple errors downstream, force its type on initialization and ignore the type for the line. For example,

```python
a: int = func() # type: ignore
# use a as an int below now
```

## 4. Commenting Style Guideline

These rules are mandatory requirements of the enforcement process. Every reviewed file must be scanned specifically for comment-style violations. 

- **Prefer self-documenting code:** Use descriptive variable/function/class names rather than comments explaining ambiguous naming conventions. Use comments to explain what variable or function names cannot describe well (behaviour / relations / edge cases)
- **Explain the why:** Where applicable, also explain why a block of code is written the way it is or the reasoning behind implementing something.
- **Module Docstrings for Large Files:** Each large file (> 300 lines of code) that handles multiple functionalities must have a top-level module docstring briefly explaining the objective of the file and what it handles.
- **Concise Docstrings for Small Functions:** Small functions should be named in a self-descriptive manner. If additional context is needed, include a brief one-line docstring.
- **Structured Docstrings for Large Functions:** Every large function (> 80 lines of code) that accepts multiple parameters must include a structured docstring in the following format:

```python
"""
<Brief Description of function>

Arguments:
  <one line brief for each argument>

Returns:
  <description of what the function returns>
"""
```

## Expected Workflow

1. Identify target files.
2. Run lint/type checks as specified in Section 3 and apply auto-fixes where safe.
3. Scan the code for comment-style compliance against Section 4 and edit directly to bring it into compliance.
4. Remove unused imports, dead code, and unreachable branches.
5. Apply logical spacing, PEP 8 practices, and JAX/Equinox fixes from Sections 1 and 2.
6. Summarize what was changed. Any pyright/ruff issue that was suppressed rather than fixed should be listed explicitly with rationale.

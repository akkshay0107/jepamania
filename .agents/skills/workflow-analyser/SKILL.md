---
name: workflow-analyser
description: Trace the logic of the codebase to find inconsistencies, poor performance, bad assumptions, or logically incorrect pieces of code. Also reviews structural/design choices and hot-path performance. Use this skill when you need to critically review an implementation.
---

# Workflow Analyser

Act as an adversarial logic tracer to find flaws, design weaknesses, and performance issues in code and architecture before they become systemic issues.

NOTE: Do not apply this to any library source code that exists in the virtual environment or the scratch directory. Editing code in virtual environments is strictly off limits. This skill reviews only — it does not edit code. Fixes are implemented only after user approval.

## 1. Logic Review

- **Logic Tracing:** Trace code execution to verify that implementation logic aligns with design. Use the scratch directory to write brute force checks for any piece of code you suspect / need to check.
- **Assumption Checking:** Challenge design and implementation assumptions. All tests passing is not a sign to stop. Reason about design ideas and try to look for simpler implementations always. Silent failure paths should be highlighted and converted into errors / intentional fallbacks where possible.
- **Inconsistency Detection:** Ensure consistency across submodules. Schemas that are shared must have validation that makes it follow the assumptions of both submodules and should not cause silent failures.

## 2. Design Review

- **Class vs. Function:** Flag classes with minimal or no state — if a class carries no meaningful instance state across its methods, or exists purely to namespace a couple of functions, recommend reducing it to plain functions.
- **Struct-like Classes:** For classes whose purpose is grouping/carrying data:
  - If purely a data container with no behavior or validation, recommend a `NamedTuple`.
  - If it has behavior, invariants, or `__post_init__` validation requirements, recommend a frozen, slotted `dataclass` (`@dataclass(frozen=True, slots=True)`) instead.
- **Optional/None Usage:** Prefer early validation over defensive `None`-checking scattered through a codepath. `None` should only be used when a value can genuinely be missing/absent — not as a stand-in for "not yet configured" or "figure it out later." Where a reasonable default exists, use it instead of `None`, and flag places where `None` is being used as a lazy placeholder rather than a true absence.
- **Early Returns / Early Errors:** Validation and error conditions should be checked and returned/raised as early as possible in a method, before any real computation happens. At the same time, flag cases where computation is being done speculatively before a return/raise that makes it unreachable — computation should happen as late as possible relative to the checks that could short-circuit it, so no cycles are wasted on a path that exits early anyway.

## 3. Performance Review

- **Hot-Path Allocation & Recomputation:** On code that exists on the hot path, look for unnecessary memory allocations and repeated computation of the same value. Recommend hoisting such values into a local variable, or caching them (e.g. `functools.lru_cache`, memoization, or an instance-level cache) when reused across calls.
- **Nested Function Overhead:** Flag nested function definitions where avoidable — a function defined inside another is reconstructed as a new function object on every call to the enclosing function. Recommend extracting to a private module- or class-level helper instead.
- **Algorithmic Improvements:** Look for heavy nested loops, excessive branching, or a data structure that's a poor fit for its access pattern (e.g. list membership checks that should be a set, linear search that should be a dict lookup). Recommend a specific alternative algorithm or data structure, not just "this is slow."
- **Vectorization Opportunities:** Identify plain Python `for` loops iterating over an array/tensor dimension that could be replaced with vectorized operations (NumPy/JAX broadcasting, `jax.vmap`, etc.).
- **CPU/GPU Sync Stalls:** On the hot path, flag operations that force a device sync or blocking transfer (e.g. `.item()`, `.block_until_ready()` called eagerly in a loop, host callbacks) where they could be deferred or batched instead.
- Think about CPU/GPU utilization as well as raw algorithmic complexity when assessing a bottleneck — a low-complexity operation that stalls the device repeatedly can be worse in practice than a higher-complexity one that stays vectorized.

## Evidence-Based Verification

Back all claims — logic, design, and performance — with verifiable evidence directly from the source code (file paths and line numbers), simple traces, or external references. Do not offer unverified claims. For performance claims specifically, write and run a benchmark script in the `scratch` directory using `uv` to confirm the bottleneck exists and to quantify the improvement of any suggested alternative before recommending it — a slowdown that seems obvious from reading the code should still be measured, not assumed.

## Expected Output

Review without directly editing code. Generate a critical review report detailing, per issue found:

1. Specific files and line numbers of the flaw (logical, design, or performance).
2. Rationale/explanation of the incorrect logic, weak design choice, or performance issue, including benchmark evidence where applicable.
3. Downstream impact of the issue.
4. Actionable recommendation to fix it.

_Note: Keep the report text-based (no massive code blocks) and implement fixes only after user approval._

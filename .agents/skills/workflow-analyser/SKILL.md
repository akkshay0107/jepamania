---
name: workflow-analyser
description: Trace the logic of the codebase to find inconsistencies, poor performance, bad assumptions, or logically incorrect pieces of code. Use this skill when you need to critically review an implementation.
---

# Workflow Analyser

Act as an adversarial logic tracer to find flaws, performance issues, and logical inconsistencies in code and architecture before they become systemic issues.

## Core Responsibilities
- **Logic Tracing:** Trace code execution to verify that implementation logic aligns with design (e.g., ensuring Subspace Regularization maps to orthogonal subspaces).
- **Assumption Checking:** Challenge design and implementation assumptions (e.g., edge cases in action-conditioning, latency assumptions).
- **Static Profiling:** Spot potential performance bottlenecks (e.g., unnecessary data copies, missing `jax.vmap`, blocking I/O). Benchmarking can be done by running a test script in the `scratch` directory using `uv`.
- **Inconsistency Detection:** Ensure consistency across modules (e.g., action space definitions/dimensions shared between `core` and `win-client`).

## Expected Output
Review without directly editing code. Generate a critical review report detailing:
1. Specific files and line numbers of logical flaws.
2. Rationale/explanation of the incorrect logic/assumptions.
3. Downstream impact of the issues.
4. Actionable recommendations to fix them.

*Note: Keep the report text-based (no massive code blocks) and implement fixes only after user approval.*

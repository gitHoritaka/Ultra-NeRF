# T45 - Port the Convex MIP Sampling and Rendering Backend

## Goal

Port the required convex MIP backend logic from the legacy repo into the
current renderer stack.

## Why This Matters

MIP support is not just a config flag. The sampling, representation, and
integration path differ from the baseline convex renderer and need a dedicated
backend implementation.

## Required Work

- Identify the legacy files/functions that implement convex MIP sampling and
  rendering.
- Port the minimum viable backend into `src/ultranerf/` in the current repo
  style.
- Keep the implementation isolated so it does not pollute the default non-MIP
  path with broad conditionals.
- Ensure convex MIP can produce usable rendered outputs and does not break
  baseline convex rendering.
- Decide how intermediate maps behave in MIP mode:
  - supported directly
  - supported partially
  - unsupported and explicitly documented

## Suggested Implementation

- Prefer a dedicated backend module or clearly separated render path.
- Reuse shared acoustic math where possible, but do not force incompatible
  abstractions.
- Avoid copying legacy debug-only code and cache machinery unless required.

## What Needs To Be Checked

- The MIP backend runs on current torch/PyTorch infrastructure.
- Baseline convex and linear rendering remain unaffected when MIP is off.
- Output shape/layout is stable and documented.

## Output of This Ticket

- Convex MIP render backend integrated into the current codebase.
- Focused tests for MIP backend shape/runtime behavior.

## Acceptance Criteria

- The current repo can render convex data using the selected MIP backend.

## Dependencies

- T44

## Blocks

- T46
- T47
- T48

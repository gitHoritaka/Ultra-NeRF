# T42 - Add Convex Dataset and Viewer Workflow Support

## Goal

Wire convex support into the current training and visualization workflows so a
user can actually run convex data through the current repo.

## Why This Matters

By this point the geometry and renderer may support convex probes, but the repo
is not complete until the user-facing entry points can select convex mode and
run end to end.

## Required Work

- Update the relevant CLI entry points to accept convex probe configs.
- Ensure training, render-demo, and viewer launchers can all initialize convex
  geometry correctly.
- Add convex-aware dataset expectations to manifests or config examples where
  needed.
- Ensure intermediate acoustic-map viewing still works in convex mode.
- Verify multi-sweep state handling still works when the scene uses convex
  probe geometry.

## Suggested Implementation

- Start with the main entry points:
  - `run_ultranerf.py`
  - `render_demo_us.py`
  - `run_visualize_sweeps.py`
  - `run_visualize_multi_sweeps.py`
- Keep the first pass focused on one coherent convex workflow rather than
  trying to update every experimental script at once.

## What Needs To Be Checked

- A convex config can be parsed from the CLI.
- Training can run in convex mode.
- The viewer can load a convex probe configuration and display the result
  correctly.
- Existing linear entry points still work unchanged.

## Output of This Ticket

- End-to-end convex-capable user workflows in the main entry points.
- Example command lines or config snippets.

## Acceptance Criteria

- A user can train, render, and visualize convex data from the current repo
  without relying on the legacy convex codebase directly.

## Dependencies

- T37
- T38
- T39
- T40
- T41

## Blocks

- T43

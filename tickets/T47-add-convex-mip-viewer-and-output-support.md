# T47 - Add Convex MIP Viewer and Output Support

## Goal

Make the current render/viewer workflows display convex MIP outputs correctly.

## Why This Matters

Even if MIP rendering works internally, the feature is incomplete if the
viewer and output adapters cannot display it correctly or distinguish it from
the baseline convex path.

## Required Work

- Update the current rendering adapters so MIP outputs reach:
  - render demo workflows
  - the single-sweep visualizer
  - the multi-sweep visualizer where relevant
- Ensure the render panel and map selector behave sensibly in MIP mode.
- Define how nearest-frame comparison should be interpreted for MIP outputs.
- Document any viewer limitations specific to MIP mode.

## Suggested Implementation

- Keep viewer support limited to what is actually reliable in the first pass.
- Prefer clear labeling when the viewer is showing an MIP render path.
- If intermediate maps differ from baseline convex mode, reflect that in the
  UI rather than silently hiding differences.

## What Needs To Be Checked

- Viewer rendering works without crashes in MIP mode.
- Output panels show the intended image/layout.
- Non-MIP viewer behavior remains unchanged.

## Output of This Ticket

- Viewer and output support for convex MIP mode.
- Manual QA notes for MIP viewer behavior.

## Acceptance Criteria

- A user can run the existing viewer path with convex MIP enabled and get
  meaningful output.

## Dependencies

- T45
- T46

## Blocks

- T48

# T54 - Add Probe Geometry and Pre-Training Visualization Confirmation

## Goal

Require the user to confirm that the selected training data is geometrically
correct before training can begin.

## Why This Matters

The todo explicitly calls for a probe-geometry entry step followed by a visual
inspection step. That is the right safeguard. Training should not start until
the user has verified that:

- the sweeps are loaded correctly
- the probe geometry is plausible
- the data is spatially aligned the way the model expects

This is especially important for convex data and mixed datasets.

## Required Work

- Add probe-geometry entry/edit controls to the training workflow.
- Support at minimum:
  - linear: width and depth
  - convex: opening angle, depth, short radius, long radius
- Feed those values into a preview visualization path.
- Launch or embed a visualization step that shows the selected sweeps with the
  chosen geometry.
- Add an explicit confirmation action indicating the data looks correct.
- Keep the final `Train` action disabled until that confirmation has happened.

## What Needs To Be Checked

- The geometry controls are understandable and match the supported probe types.
- The preview reflects the selected geometry rather than stale dataset values.
- The training workflow cannot skip the confirmation step accidentally.
- The preview is good enough to catch obvious geometry-placement mistakes.

## Output of This Ticket

- Probe-geometry input UI in the training workflow.
- A pre-training visualization confirmation step.
- Gating logic that enables training only after confirmation.

## Acceptance Criteria

- A user can input probe geometry, preview the data placement, and explicitly
  confirm it before training.
- Training remains disabled until confirmation is complete.

## Dependencies

- T52
- T53

## Blocks

- T55
- T57

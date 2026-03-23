# T37 - Add Convex Probe Configuration and CLI Support

## Goal

Add clean convex-probe configuration support to the current config and CLI
paths.

## Why This Matters

The legacy convex repo exposes many convex-specific parameters through its own
argument parser, then mutates a global configuration object. The current repo
has a cleaner `configargparse`-style flow and should keep that style.

If convex parameters are not integrated cleanly, later geometry and rendering
changes will become difficult to validate and easy to misconfigure.

## Required Work

- Add a `probe_type` option to the current config path.
- Add the convex parameters selected in T36 to the current argument parser and
  config files.
- Ensure those values flow into the current runtime objects without a global
  mutable singleton.
- Keep current linear-probe behavior unchanged when `probe_type=linear`.
- Add validation for invalid convex configurations such as:
  - missing required radii/angles
  - negative or zero radii
  - fan angle outside a sensible range
- Decide which entry points need convex support immediately:
  - baseline training
  - render demo
  - visualizer launchers

## Suggested Implementation

- Extend the current config utilities in `src/ultranerf/unerf_config.py`.
- Keep defaults compatible with existing linear configs.
- Group convex-specific arguments together and document them clearly.
- Avoid porting legacy flags that are not required for the first pass.

## What Needs To Be Checked

- Existing linear configs still parse and run unchanged.
- Convex-specific arguments reach the geometry/ray-generation code paths.
- The config structure is shared consistently between training, rendering, and
  visualization.

## Output of This Ticket

- Updated CLI/config support for convex probes.
- Validation helpers for convex parameter sanity.
- Example convex config file or config snippet.

## Acceptance Criteria

- A user can select `probe_type=convex` from the current codebase without
  touching legacy files.
- Linear configs remain backward compatible.

## Dependencies

- T36

## Blocks

- T38
- T39
- T42
- T43

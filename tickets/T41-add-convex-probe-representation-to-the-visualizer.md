# T41 - Add Convex Probe Representation to the Visualizer

## Goal

Represent convex probes correctly in the viewer instead of reusing the current
linear rectangular probe overlay.

## Why This Matters

Even if convex data is fused and rendered correctly, the visualizer will still
be misleading if the interactive probe overlay shows a linear scan plane and
beam model.

The viewer needs a convex-specific visual representation that makes the fan
shape and orientation obvious.

## Required Work

- Add a convex probe overlay representation:
  - fan/sector support
  - probe axes
  - center/radial direction
  - beam/fan boundaries
- Keep the current linear representation intact.
- Ensure the visualizer chooses the correct representation from probe
  geometry/config.
- Verify the convex probe overlay aligns with convex sweep fusion in 3D.

## Suggested Implementation

- Extend `src/ultranerf/visualization/probe_representation.py`.
- Reuse the current napari layer pattern where appropriate.
- Keep the overlay minimal and readable; do not import all legacy debug
  overlays.

## What Needs To Be Checked

- Linear viewers still show the existing overlay behavior.
- Convex viewers show a clear fan-sector probe representation.
- The overlay orientation matches the actual convex mapping used for fusion and
  rendering.

## Output of This Ticket

- Convex-aware probe overlays in the visualizer.
- Tests for probe-representation geometry.

## Acceptance Criteria

- A user can visually distinguish linear and convex probe geometry in the
  viewer.
- The convex overlay aligns with the fused data and probe pose.

## Dependencies

- T36
- T40

## Blocks

- T42
- T43

# T39 - Integrate Convex Rendering Layout Into the Active Renderer

## Goal

Make the active renderer produce valid convex outputs, including whatever
output reshaping or remapping is required to interpret convex images
correctly.

## Why This Matters

Convex support is not finished when rays exist. The renderer also has to know
how to interpret the per-ray outputs in image space.

The legacy convex repo includes additional output-layout handling and remapping
logic. Some of that is required for correctness, even if not all of the
legacy rendering branches are merged.

## Required Work

- Identify the minimum renderer changes needed so convex outputs are displayed
  correctly.
- Port only the necessary layout/remapping logic from the legacy repo.
- Decide what the primary output should be for the first pass:
  - native convex fan layout
  - remapped rectangular image
  - both, if that is low risk
- Ensure the currently exposed intermediate acoustic maps still work in convex
  mode.
- Keep the current linear rendering output unchanged.

## Suggested Implementation

- Keep the acoustic integration logic shared when possible.
- Isolate convex-specific output layout handling near the render adapter rather
  than scattering conditionals across unrelated code.
- Defer legacy extras such as MIP and full-volume mode unless they are proven
  necessary for baseline convex rendering.

## What Needs To Be Checked

- Convex rendering produces a geometrically meaningful image.
- The output shape is consistent with the chosen convex image convention.
- Intermediate maps remain accessible through the current render panel API.

## Output of This Ticket

- Convex-compatible rendering output path.
- Tests for convex render shape/layout behavior.

## Acceptance Criteria

- The active renderer can return usable convex images and intermediate maps.
- Linear rendering behavior remains unchanged.

## Dependencies

- T38

## Blocks

- T42
- T43

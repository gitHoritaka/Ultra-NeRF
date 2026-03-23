# T38 - Add Convex Ray Generation for Training and Inference

## Goal

Add convex ray generation to the active training and inference pipeline.

## Why This Matters

The current renderer path is built around `get_rays_us_linear(...)`. Convex
support becomes real only when the model can generate and sample rays using a
convex probe geometry.

The legacy convex repo already contains convex-specific ray-generation logic,
including fan-based and concentric/radial sampling strategies. That logic
needs to be ported selectively into the current codebase.

## Required Work

- Refactor the current ray-generation entry point so it dispatches by
  `probe_type`.
- Port the minimum viable convex ray-generation logic from the legacy repo.
- Ensure the output format is compatible with the current rendering path:
  - ray origins
  - ray directions
  - sample counts / layout expectations
- Decide whether the first pass supports:
  - a single convex sampling strategy
  - or the minimal subset of the legacy strategy options
- Keep the current linear path unchanged and verified.
- Add shape and sanity validation for convex rays:
  - angular spread
  - radial direction consistency
  - origin placement

## Suggested Implementation

- Start from the legacy functions conceptually, but rewrite them into the
  current codebase style rather than copying them verbatim.
- Keep the public entry point near `src/ultranerf/nerf_utils.py`.
- Keep geometry parameters explicit and typed.

## What Needs To Be Checked

- Linear ray generation still produces the same results as before.
- Convex rays have the expected fan geometry in probe-local coordinates.
- The active renderer can consume the convex ray bundle without shape errors.

## Output of This Ticket

- Geometry-dispatched ray generation for both linear and convex probes.
- Unit tests for convex ray generation.

## Acceptance Criteria

- Training and inference can request convex rays through the current codebase.
- No legacy global convex config is required.

## Dependencies

- T36
- T37

## Blocks

- T39
- T42
- T43

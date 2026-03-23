# T36 - Define the Convex Probe Geometry Contract

## Goal

Define the geometry model that the current codebase will use for convex
probes.

This ticket is the design boundary between the current linear-only
implementation and a future geometry-aware implementation that supports both
linear and convex probes without duplicating the whole pipeline.

## Why This Matters

The current code assumes a linear probe in multiple places:

- linear ray generation in `src/ultranerf/nerf_utils.py`
- rectangular pixel-to-probe mapping in
  `src/ultranerf/visualization/transforms.py`
- rectangular probe visualization in
  `src/ultranerf/visualization/probe_representation.py`
- linear sweep fusion assumptions in
  `src/ultranerf/visualization/sweep_volume.py`

The legacy convex codebase adds convex support, but it does so through a
separate path with global mutable configuration, duplicated helper logic, and
extra experimental branches that should not all be merged into the current
repo.

Before porting code, the current repo needs one explicit contract for convex
geometry.

## Required Work

- Define a canonical probe geometry model that supports at least:
  - `probe_type`
  - linear width/depth
  - convex fan center
  - convex fan angle
  - convex inner and outer radius, or equivalent radius model
  - optional anisotropic scaling if the legacy implementation requires it
- Decide which convex parameters are truly required in the current repo and
  which legacy parameters are experimental and should be deferred.
- Define the probe-local coordinate frame for convex probes:
  - where the fan center sits
  - which axis is depth/radial direction
  - how lateral angle is defined
- Define the mapping contract for convex data:
  - image pixel -> probe-local point
  - probe-local sample -> world-space point
  - world pose -> rendered probe orientation
- Define how convex output image layout is represented in the renderer:
  - native fan layout
  - remapped rectangular image
  - both, if needed
- Decide which parts of the legacy convex implementation are in scope for the
  first merge:
  - required geometry and sampling
  - required render/output reshaping
  - not yet MIP mode
  - not yet full-volume mode
  - not yet legacy cache/debug branches

## Suggested Implementation

- Add a geometry-facing typed structure in the current codebase rather than
  importing the legacy global `convex.config`.
- Extend the existing probe geometry structures under
  `src/ultranerf/visualization/transforms.py` or a nearby geometry module.
- Document one shared probe-local convention to be used by:
  - training
  - rendering
  - visualization
  - sweep fusion

## What Needs To Be Checked

- The geometry contract can express both linear and convex probes.
- The chosen convex parameters correspond to the legacy implementation closely
  enough that its ray-generation logic can be ported without ambiguity.
- The contract is explicit about whether rendered images stay in fan space or
  are remapped into a rectangular image grid.

## Output of This Ticket

- A documented convex geometry contract.
- Updated probe-geometry data structures in the current repo.
- A written decision about which legacy convex features are in scope for the
  first integration pass.

## Acceptance Criteria

- A developer can implement convex support in the current repo without
  depending on the legacy global configuration pattern.
- The distinction between linear and convex probe-local geometry is explicit.
- The image-layout convention for convex rendering is defined.

## Dependencies

- T01

## Blocks

- T37
- T38
- T39
- T40
- T41
- T42
- T43

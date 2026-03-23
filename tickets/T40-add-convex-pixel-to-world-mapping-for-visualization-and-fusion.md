# T40 - Add Convex Pixel-to-World Mapping for Visualization and Fusion

## Goal

Extend the visualization and sweep-fusion pipeline so convex frames are mapped
into world space correctly.

## Why This Matters

The current visualizer assumes each ultrasound image is a rectangular linear
scan plane. Convex probes instead sample a fan sector. If the visualization
keeps using linear rectangular mapping, the fused 3D volume and scan-plane
display will be wrong even if training/inference supports convex probes.

## Required Work

- Generalize image-pixel to probe-local mapping to support convex fan geometry.
- Generalize scan-support geometry from a rectangle to a fan sector or another
  correct convex representation.
- Update sweep fusion so sampled convex image points map into the right
  world-space positions.
- Update bounds estimation for convex sweep volumes.
- Decide how fan-space masking/background should be handled in fusion:
  - skip pixels outside the fan
  - or explicitly define a valid-mask path

## Suggested Implementation

- Extend the existing transform utilities under
  `src/ultranerf/visualization/transforms.py`.
- Keep the mapping logic shared where possible, but allow probe-type-specific
  implementations.
- Ensure the sweep-fusion backend can operate without needing UI code.

## What Needs To Be Checked

- Linear pixel mapping still behaves exactly as before.
- Convex pixel mapping produces correct fan geometry in probe-local space.
- Convex sweep fusion no longer treats the image as a flat rectangle.

## Output of This Ticket

- Convex-aware transform utilities.
- Convex-aware sweep fusion and bounds estimation.
- Unit tests for mapping correctness.

## Acceptance Criteria

- Convex sweeps can be fused and visualized in world space without rectangular
  distortion.

## Dependencies

- T36

## Blocks

- T41
- T42
- T43

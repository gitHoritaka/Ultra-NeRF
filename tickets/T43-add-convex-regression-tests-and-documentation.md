# T43 - Add Convex Regression Tests and Documentation

## Goal

Add regression coverage and concise user/developer documentation for convex
probe support.

## Why This Matters

Convex support changes geometry assumptions across training, rendering,
visualization, and sweep fusion. Without tests and clear documentation, the
feature will be fragile and hard to maintain.

## Required Work

- Add targeted tests for:
  - convex config parsing
  - convex ray generation
  - convex render output shape/layout
  - convex pixel-to-world mapping
  - convex probe representation
  - convex viewer state wiring
- Add at least one smoke test for the main convex entry path.
- Document:
  - supported convex parameters
  - current limitations
  - chosen image-layout convention
  - differences from the legacy convex repo
- Document which legacy convex features were intentionally deferred.

## Suggested Implementation

- Keep tests small and geometry-focused.
- Add concise README links and place detailed notes under `docs/`.
- Make it clear that convex support was selectively ported, not wholesale
  copied from the legacy codebase.

## What Needs To Be Checked

- Convex support has enough regression coverage to catch shape/layout breaks.
- Docs are aligned with the current implementation, not the legacy repo.
- Linear coverage remains intact.

## Output of This Ticket

- Convex regression tests.
- Convex support documentation for developers and users.

## Acceptance Criteria

- A developer can understand how convex support works and what remains out of
  scope.
- The repo has automated coverage for the critical convex geometry paths.

## Dependencies

- T36
- T37
- T38
- T39
- T40
- T41
- T42

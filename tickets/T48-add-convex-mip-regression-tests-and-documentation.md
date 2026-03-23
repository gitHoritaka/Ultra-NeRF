# T48 - Add Convex MIP Regression Tests and Documentation

## Goal

Add regression coverage and concise documentation for the migrated convex MIP
path.

## Why This Matters

MIP is an optional advanced path. That makes it especially easy to break if it
is not documented and tested explicitly.

## Required Work

- Add tests for:
  - MIP config parsing
  - backend selection
  - MIP render shape/layout
  - viewer/output adapter behavior
- Add at least one smoke test for the MIP-enabled render path.
- Document:
  - how to enable convex MIP
  - what it changes compared with baseline convex rendering
  - known limitations
  - what was intentionally not ported from legacy MIP code

## Suggested Implementation

- Keep test coverage focused on contract stability, not numerical perfection.
- Put detailed notes in `docs/` and add only a short README link.
- Be explicit that MIP is an optional advanced mode.

## What Needs To Be Checked

- MIP mode has automated coverage.
- Docs reflect the current implementation rather than legacy behavior.
- Baseline convex docs remain clear and separate.

## Output of This Ticket

- Regression tests for convex MIP support.
- User/developer docs for the MIP path.

## Acceptance Criteria

- A developer can understand how to use and maintain convex MIP support in the
  current repo.

## Dependencies

- T44
- T45
- T46
- T47

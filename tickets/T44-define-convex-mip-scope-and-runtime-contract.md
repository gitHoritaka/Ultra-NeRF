# T44 - Define the Convex MIP Scope and Runtime Contract

## Goal

Define exactly what "MIP support" means in the current repo before porting any
legacy convex MIP code.

## Why This Matters

The legacy convex repo contains additional MIP-related code, but it is mixed
with other experimental branches. Porting it without a clear scope would make
the current renderer harder to reason about and harder to keep stable.

The current repo first needs a written contract covering:

- which MIP behavior is being migrated
- where it plugs into the current runtime
- what remains optional or out of scope

## Required Work

- Inspect the legacy convex MIP path and identify the minimum viable feature
  set to migrate.
- Define whether MIP support is:
  - a separate renderer mode
  - a sampling mode inside the active renderer
  - or an alternate render backend selected by config
- Define how MIP mode interacts with:
  - convex geometry
  - intermediate acoustic maps
  - render output layout
  - training vs inference
- Decide which legacy MIP branches remain out of scope for the first pass.
- Define compatibility expectations for the non-MIP convex path.

## Suggested Implementation

- Write this as a short design note before code changes.
- Keep MIP-specific settings clearly separated from baseline convex settings.
- Prefer one explicit runtime switch rather than implicit behavior.

## What Needs To Be Checked

- The scope is small enough to be implemented without destabilizing the active
  renderer.
- The repo still has one clear default path for non-MIP convex rendering.
- A developer can tell exactly when MIP is active and what code path is used.

## Output of This Ticket

- A documented MIP runtime contract.
- A written in-scope / out-of-scope decision list for legacy MIP features.

## Acceptance Criteria

- The repo has an explicit MIP plan before any MIP backend code is merged.

## Dependencies

- T39

## Blocks

- T45
- T46
- T47
- T48

# T46 - Add Convex MIP Configuration and CLI Wiring

## Goal

Expose the convex MIP path through the current CLI/config system.

## Why This Matters

Once the MIP backend exists, users still need a clean and explicit way to
enable it. That should happen through the current repo's config flow, not via
legacy mutable globals.

## Required Work

- Add explicit MIP selection flags/config fields to the current config path.
- Define sane defaults so baseline linear and convex runs remain unchanged.
- Ensure incompatible configurations are rejected clearly, for example:
  - MIP selected without convex support if that is unsupported
  - unsupported combinations of MIP and viewer/output settings
- Wire the selected config into the runtime backend selection.
- Update example configs or commands.

## Suggested Implementation

- Extend `src/ultranerf/unerf_config.py`.
- Keep MIP settings grouped and documented.
- Use explicit booleans or a render-mode enum rather than implicit side
  effects.

## What Needs To Be Checked

- Existing configs continue to work unchanged.
- A user can select MIP mode intentionally and see the correct backend used.
- Invalid combinations fail early with understandable errors.

## Output of This Ticket

- Config/CLI support for convex MIP rendering.
- Example MIP config snippet.

## Acceptance Criteria

- Convex MIP can be enabled through the current repo's CLI/config system.

## Dependencies

- T44
- T45

## Blocks

- T47
- T48

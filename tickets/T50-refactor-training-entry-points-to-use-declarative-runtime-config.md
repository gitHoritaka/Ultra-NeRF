# T50 - Refactor Training Entry Points to Use Declarative Runtime Config

## Goal

Refactor the current training commands so they consume the training config
contract from T49 instead of relying on script-local defaults and ad hoc
arguments.

## Why This Matters

The config contract is only useful if the real training paths honor it. The
repo currently has multiple training entry points, and they should resolve
training behavior from the same declarative source rather than each script
making its own assumptions.

This ticket is the implementation bridge between "we wrote down the config
contract" and "training actually follows it."

## Required Work

- Identify the active training entry points that should adopt the new contract.
- Add config loading/parsing that resolves:
  - core runtime settings
  - dataset and probe geometry
  - optimization settings
  - validation/checkpoint cadence
  - output layout
- Remove or reduce hard-coded defaults where they conflict with the new
  contract.
- Keep a small set of explicit CLI overrides where useful, but make their
  precedence predictable and documented.
- Ensure the resolved runtime configuration can be inspected or logged at run
  start.
- Preserve backwards-compatible behavior where feasible, or fail with readable
  upgrade messages.

## What Needs To Be Checked

- Existing training flows still launch with the new config path.
- Resolved training settings match the config file rather than hidden defaults.
- CLI overrides behave consistently and are easy to explain.
- Output directories and saved runtime args reflect the resolved configuration.

## Output of This Ticket

- Refactored training entry points that read from the declarative config.
- Shared config-loading logic rather than duplicated parsing in each script.
- Tests or smoke coverage for config resolution.

## Acceptance Criteria

- A developer can change step count or similar training behavior in config and
  see it take effect without editing Python.
- Runtime logs show the resolved training configuration.
- The main training scripts use the same config semantics.

## Dependencies

- T49

## Blocks

- T51
- T52
- T55

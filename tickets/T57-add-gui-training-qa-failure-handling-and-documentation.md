# T57 - Add GUI Training QA, Failure Handling, and Documentation

## Goal

Harden the GUI training workflow with documentation, QA coverage, and explicit
handling of common failure modes.

## Why This Matters

A training GUI touches file discovery, geometry setup, preview rendering,
background execution, and progress reporting. That is too much surface area to
ship without written workflow documentation and practical QA guidance.

This ticket should close the loop after the interactive pieces are working.

## Required Work

- Document the end-to-end GUI training workflow in `README` and/or `docs/`.
- Add a manual QA checklist that covers:
  - discovery
  - training/validation selection
  - geometry confirmation
  - scheme selection
  - training launch
  - validation-preview updates
- Add tests where realistic, especially around:
  - state transitions
  - disabled/enabled training button logic
  - failure handling
- Handle common failure cases explicitly, for example:
  - missing or invalid sweeps
  - invalid geometry input
  - failed training process start
  - missing validation preview outputs
  - background-process crash
- Define what artifacts the GUI should preserve for debugging failed runs.

## What Needs To Be Checked

- A new user can follow the documented workflow without reverse-engineering the
  code.
- Common failure states do not leave the GUI in a confusing partial state.
- The training GUI can be regression-tested at least at the controller/state
  level.

## Output of This Ticket

- Documentation for the GUI training workflow.
- QA checklist and failure-handling behavior.
- Regression coverage for the training GUI state machine where feasible.

## Acceptance Criteria

- The GUI training workflow is documented and testable.
- Failure states are visible and recoverable.
- The feature can be handed to another developer without extra tribal
  knowledge.

## Dependencies

- T52
- T53
- T54
- T55
- T56

## Blocks

- None

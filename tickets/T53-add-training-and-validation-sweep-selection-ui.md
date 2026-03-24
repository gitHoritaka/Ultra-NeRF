# T53 - Add Training and Validation Sweep Selection UI

## Goal

Let the user assign discovered sweeps to training and validation roles from the
GUI workflow.

## Why This Matters

The training GUI is not useful if it only discovers sweeps. The user needs to
choose:

- which sweeps participate in training
- which sweep or sweeps are reserved for validation

This ticket should make those roles explicit and hard to confuse.

## Required Work

- Extend the training setup window with training/validation assignment UI.
- Let the user select:
  - one or more training sweeps
  - one validation sweep or validation set, depending on supported policy
- Make the selected roles visible and easy to audit before continuing.
- Prevent invalid combinations, such as:
  - zero training sweeps
  - the same sweep accidentally assigned twice if that is not allowed
- Surface enough context next to each sweep to support the choice.
- Persist the selection in the pending training session state.

## What Needs To Be Checked

- Sweep role assignment is clear and does not depend on implied defaults.
- Invalid training/validation combinations are blocked or explained.
- The selected set is preserved when the user moves to later setup steps.

## Output of This Ticket

- GUI controls for assigning sweeps to training and validation roles.
- Validation logic for invalid or incomplete selections.

## Acceptance Criteria

- A user can select training and validation sweeps from the discovered list.
- The session state clearly reflects those selections before training starts.

## Dependencies

- T52

## Blocks

- T54
- T55
- T57

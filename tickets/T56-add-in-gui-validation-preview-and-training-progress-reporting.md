# T56 - Add In-GUI Validation Preview and Training Progress Reporting

## Goal

Show training progress in the GUI, including periodic validation renders from
sample validation poses.

## Why This Matters

The todo explicitly calls for background training with validation previews shown
in the GUI during validation iterations. Without this, the user can launch
training but still has no quick way to judge whether the run is healthy.

## Required Work

- Add a live progress area to the training workflow or a related monitoring
  window.
- Show key status information such as:
  - current step
  - total steps
  - latest training loss
  - latest validation step
  - output directory
- Define how validation preview images are selected, for example:
  - one fixed sample pose from the validation sweep
  - a small stable validation subset
- Display the latest validation render in the GUI whenever a validation pass
  completes.
- Update the progress UI from the background training process safely.
- Handle completion, cancellation, and failure states cleanly.

## What Needs To Be Checked

- The progress UI updates while training is running.
- Validation previews correspond to the intended validation selection.
- Completion and failure states are clearly visible.
- The UI can handle long runs without leaking widgets or stale sessions.

## Output of This Ticket

- In-GUI training progress reporting.
- Validation-preview rendering during training.
- Completion/failure state handling for the active training session.

## Acceptance Criteria

- A running training job reports progress in the GUI.
- Validation preview images appear during validation iterations.
- The user can tell whether the run is healthy without inspecting shell logs.

## Dependencies

- T55

## Blocks

- T57

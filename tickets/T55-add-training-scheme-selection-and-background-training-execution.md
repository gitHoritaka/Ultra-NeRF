# T55 - Add Training-Scheme Selection and Background Training Execution

## Goal

Allow the user to choose a training scheme from the GUI and start training in a
background process without freezing the app.

## Why This Matters

This is the point where the training GUI becomes operational. The user should
be able to:

- pick a training scheme from a dropdown
- start training from the prepared session state
- keep using the GUI while training runs

This ticket depends on the declarative training config and scheme work so the
GUI launches a real, reproducible training recipe rather than a hand-built
callback.

## Required Work

- Add a training-scheme dropdown driven by the scheme definitions from T51.
- Show enough scheme context for the user to understand what is being chosen.
- Resolve the final training runtime configuration from:
  - selected dataset/sweeps
  - validation selection
  - probe geometry
  - chosen training scheme
- Start training in the background rather than blocking the main GUI thread.
- Define how background execution is managed, for example:
  - worker process
  - subprocess
  - queued task runner
- Surface startup failures clearly in the GUI.
- Persist the launched training session metadata so later preview/progress
  updates know which run is active.

## What Needs To Be Checked

- Starting training does not freeze the viewer.
- The chosen scheme actually affects the launched config.
- The output run directory and saved config are inspectable.
- Background-process failure is visible and actionable.

## Output of This Ticket

- A training-scheme selector in the GUI.
- Background training execution.
- Resolved training-run metadata and launch logging.

## Acceptance Criteria

- A user can pick a scheme and start training from the GUI.
- The app remains responsive while training runs.
- The launched run is reproducible from saved config artifacts.

## Dependencies

- T51
- T53
- T54

## Blocks

- T56
- T57

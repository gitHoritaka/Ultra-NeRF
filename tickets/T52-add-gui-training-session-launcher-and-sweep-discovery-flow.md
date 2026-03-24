# T52 - Add a GUI Training Session Launcher and Sweep Discovery Flow

## Goal

Add the first GUI entry point for training an UltraNeRF model from within the
application.

The initial scope is discovery and session setup, not yet actual training.

## Why This Matters

The desired workflow starts from a training button in the GUI. Clicking that
button should open a dedicated training-preparation window instead of forcing
the user back to shell scripts.

Before the GUI can collect sweep roles or training schemes, it must first:

- let the user choose a root folder
- discover compatible sweeps
- present them in a form the user can inspect

## Required Work

- Add a visible training entry point in the GUI.
- Open a dedicated training setup window or dialog rather than overloading the
  current visualization controls.
- Allow the user to choose a folder containing candidate sweeps.
- Search that folder for datasets compatible with the repo's expected training
  formats.
- Present a discovered-sweeps list that includes enough metadata to make a
  decision, for example:
  - sweep id or folder name
  - frame count
  - image shape
  - whether poses are available
  - probe type if known
- Define and display clear failure states when no compatible sweeps are found.

## What Needs To Be Checked

- The training launcher is easy to find from the GUI.
- Dataset discovery works on realistic folder structures.
- Incompatible folders fail clearly instead of silently.
- The discovery output is stable enough for later selection steps.

## Output of This Ticket

- A training setup window.
- Folder selection and sweep discovery.
- A compatible-sweeps list in the GUI.

## Acceptance Criteria

- A user can open the training workflow and point it at a sweep root.
- The GUI lists discovered compatible sweeps without starting training yet.

## Dependencies

- T49
- T50
- T51

## Blocks

- T53
- T54
- T55
- T57

# T51 - Add Config-Driven Training Schemes and Regression Coverage

## Goal

Introduce explicit training-scheme files or presets that capture loss
composition and other recipe-level training behavior, and add regression
coverage around that system.

## Why This Matters

The GUI todo explicitly calls for a dropdown of training schemes. That dropdown
must point to something real and stable. A "training scheme" should not just be
marketing text; it should resolve to concrete, versioned behavior such as:

- which losses are active
- how those losses are weighted
- any schedule-related differences that define the recipe

Without this ticket, the GUI would have nothing robust to select.

## Required Work

- Decide how training schemes are stored:
  - separate config files
  - structured presets
  - or another explicit, versioned representation
- Implement loading and validation of those schemes.
- Wire the scheme into the training runtime so loss selection and weighting are
  driven from the chosen scheme.
- Add a small initial set of supported schemes with clear names and intent.
- Ensure the resolved scheme is recorded in training outputs.
- Add regression tests for:
  - valid scheme loading
  - missing or invalid scheme references
  - resolved loss weights reaching the training runtime

## What Needs To Be Checked

- A scheme can be selected without editing code.
- Training logs record which scheme ran.
- Invalid schemes fail early with readable errors.
- Existing default training behavior has an equivalent scheme so the repo does
  not regress functionally.

## Output of This Ticket

- A training-scheme mechanism.
- One or more reusable training-scheme definitions.
- Tests that protect against future config/runtime drift.

## Acceptance Criteria

- The repo has a concrete notion of a training scheme.
- A later GUI dropdown can list supported schemes directly.
- Loss behavior no longer depends on hidden script branches.

## Dependencies

- T49
- T50

## Blocks

- T52
- T55
- T57

# T49 - Define the Training Config and Loss-Scheme Contract

## Goal

Define a stable configuration contract for UltraNeRF training so training
behavior is controlled by config files rather than scattered hard-coded script
defaults.

The contract should also define how a training run references the file or
preset that governs which losses are active and how they are combined.

## Why This Matters

Current training behavior is still too script-driven. Important runtime choices
such as step count, validation cadence, learning rates, and loss composition
are hard to inspect, hard to reproduce, and hard for a GUI workflow to drive
reliably.

Before adding any training GUI, the repo needs one clear answer to:

- which values belong in the main training config
- how loss schemes are referenced
- how script defaults are overridden
- what is considered part of a reproducible training recipe

## Required Work

- Inspect the current training entry points and list the parameters that
  materially affect training behavior.
- Define the minimum required contents of a training config, including:
  - dataset path or manifest path
  - probe type and geometry
  - render mode
  - step count
  - batch sizing and sampling settings
  - optimization settings
  - validation cadence
  - checkpoint cadence
  - output directory
- Define how loss behavior is selected, for example:
  - a referenced loss-scheme file
  - a named preset
  - or an inline structured block
- Define which values remain runtime-only CLI overrides and which must live in
  config for reproducibility.
- Write down compatibility expectations for existing training configs.

## Suggested Implementation

- Write a short design note or ticket-local contract description first.
- Keep the loss-scheme reference explicit rather than hidden in a script
  branch.
- Prefer a config structure that a GUI can populate without reverse-engineering
  Python defaults.

## What Needs To Be Checked

- A developer can tell from one config which training behavior will run.
- Loss selection is explicit and inspectable.
- The contract can support both current linear/convex paths and later MIP
  variants.
- The contract is stable enough for a GUI to rely on.

## Output of This Ticket

- A documented training config contract.
- A documented loss-scheme contract.
- A list of the currently supported runtime fields and their intended meaning.

## Acceptance Criteria

- The repo has one explicit design for config-driven training behavior.
- A loss scheme can be referenced without reading training script internals.
- Later implementation tickets can target this contract directly.

## Dependencies

- None

## Blocks

- T50
- T51
- T52
- T55

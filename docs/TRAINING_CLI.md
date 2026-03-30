# CLI Training

This document covers direct training from the terminal without the GUI.

## Main Entry Point

Use:

```bash
python run_ultranerf.py --config <config.txt>
```

Example:

```bash
python run_ultranerf.py --config configs/config_base_nerf.txt
```

## Config Structure

Training uses:

- a flat config file for dataset, probe, and runtime fields
- an optional JSON `training_scheme` for reusable recipe behavior

The flat config carries fields such as:

- `datadir`
- `split_file`
- `probe_type`
- `probe_width`
- `probe_depth`
- `basedir`
- `expname`

The JSON training scheme carries fields such as:

- `n_iters`
- logging / checkpoint cadence
- validation preview cadence
- loss terms and weights
- regularization toggles

See:

- [`TRAINING_CONFIG.md`](TRAINING_CONFIG.md)
- [`../configs/training_schemes/`](../configs/training_schemes)

## Typical Linear Example

```ini
expname = spine_linear_example
basedir = ./logs
datadir = ./data/my_dataset
split_file = ./data/my_dataset/split.json
dataset_type = us
probe_type = linear
probe_width = 80
probe_depth = 140
training_scheme = ./configs/training_schemes/l2_baseline.json
```

Run:

```bash
python run_ultranerf.py --config /path/to/config.txt
```

## Typical Convex Example

Use a convex config with the required fan parameters:

- `probe_type = convex`
- `convex_center_x`
- `convex_center_y`
- `convex_angle_deg`
- `convex_inner_radius_px`
- `convex_outer_radius_px`
- `convex_scale_x_mm`
- `convex_scale_y_mm`
- `convex_n_rays`
- `convex_n_samples`

If you want the MIP path, also set:

- `render_mode = convex_mip`

See:

- [`CONVEX_SUPPORT.md`](CONVEX_SUPPORT.md)
- [`CONVEX_MIP_SUPPORT.md`](CONVEX_MIP_SUPPORT.md)

## Outputs

Each run writes under:

- `logs/<expname>/`

Common outputs:

- `args.txt`
- `config.txt` if a source config file was provided
- `resolved_training_config.json`
- checkpoints such as `001000.tar`
- `progress.jsonl`
- `validation_preview/latest.gif`

## Validation Preview

When validation indices are available and `validation_preview_every > 0`,
training writes an animated validation summary:

- several sampled validation poses
- each frame shows `ground truth | render`
- the latest animation is saved at `validation_preview/latest.gif`

## Notes

- `load_us.py` expects training images in a loader-friendly `0..255` range and
  normalizes them internally to `[0, 1]`.
- The GUI training workflow prepares datasets for that assumption automatically.
- If you prepare datasets manually, keep the same expectation in mind.

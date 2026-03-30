# Training And Viewer Recipes

This file is the shortest path for the common workflows:

- train a linear model
- train a convex model
- train a convex MIP model
- open any trained checkpoint in the viewer later

## 1. Train A Linear Model

Prepare a dataset directory with:

- `images.npy`
- `poses.npy`

Then run:

```bash
python run_ultranerf.py --config configs/config_base_nerf.txt
```

Or make an explicit linear config:

```ini
expname = my_linear_run
basedir = ./logs
datadir = ./data/my_linear_dataset
dataset_type = us
probe_type = linear
probe_width = 80
probe_depth = 140
training_scheme = ./configs/training_schemes/l2_baseline.json
```

and run:

```bash
python run_ultranerf.py --config /path/to/linear_config.txt
```

## 2. Train A Convex Model

If the source dataset is not already in UltraNeRF format, convert it first:

```bash
python convert_convex_dataset.py \
  --source-root /workspace/data/abdominal_phantom \
  --output-root /workspace/data/abdominal_phantom_ultranerf
```

Then use a convex config with:

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

Example:

```bash
python run_ultranerf.py --config configs/config_convex_example.txt
```

## 3. Train A Convex MIP Model

Use a convex config and set:

- `render_mode = convex_mip`

For the current abdominal example:

```bash
python run_ultranerf.py --config configs/config_abdominal_phantom_convex_mip_v4.txt
```

Optional MIP fields:

- `mip_use_elongation`
- `mip_max_elongation`
- `mip_pixel_radius`

## 4. Train From The GUI

Launch:

```bash
python run_training_gui.py
```

Then:

1. Choose the sweep root.
2. Discover sweeps.
3. Set train and validation roles.
4. Set the probe geometry.
5. Open the preview and confirm it.
6. Choose a training scheme.
7. Start training.
8. After completion, click `Open Result in Viewer`.

The GUI training path prepares a combined dataset automatically and writes the
run under `logs/gui_training/<run_id>/`.

## 5. Open A Trained Model In The Viewer

The viewer entry point is:

```bash
python run_visualize_multi_sweeps.py --manifest-path <manifest.json>
```

To use a checkpoint, add:

- `--checkpoint-path`
- `--config-path`

Example:

```bash
python run_visualize_multi_sweeps.py \
  --manifest-path data/spine_phantom/multi_sweep_manifest.json \
  --checkpoint-path logs/spine_linear_example/003000.tar \
  --config-path logs/spine_linear_example/args.txt \
  --device cuda \
  --render-trigger-mode manual
```

The same viewer command works for:

- linear checkpoints
- convex checkpoints
- convex MIP checkpoints

The probe type and render mode are taken from the checkpoint config.

## 6. Current Convex Example

Latest corrected abdominal convex MIP example:

- manifest:
  [`data/abdominal_phantom_ultranerf/multi_sweep_manifest.json`](../data/abdominal_phantom_ultranerf/multi_sweep_manifest.json)
- checkpoint:
  [`logs/abdominal_phantom_convex_mip_v4/best_checkpoint.tar`](../logs/abdominal_phantom_convex_mip_v4/best_checkpoint.tar)
- config:
  [`logs/abdominal_phantom_convex_mip_v4/args.txt`](../logs/abdominal_phantom_convex_mip_v4/args.txt)

Run:

```bash
python run_visualize_multi_sweeps.py \
  --manifest-path data/abdominal_phantom_ultranerf/multi_sweep_manifest.json \
  --checkpoint-path logs/abdominal_phantom_convex_mip_v4/best_checkpoint.tar \
  --config-path logs/abdominal_phantom_convex_mip_v4/args.txt \
  --device cuda \
  --render-trigger-mode manual \
  --fusion-device auto
```

## 7. Which Docs To Read Next

- linear/convex CLI training details:
  [`TRAINING_CLI.md`](TRAINING_CLI.md)
- config fields and training schemes:
  [`TRAINING_CONFIG.md`](TRAINING_CONFIG.md)
- standalone training GUI:
  [`GUI_TRAINING_WORKFLOW.md`](GUI_TRAINING_WORKFLOW.md)
- convex runtime details:
  [`CONVEX_SUPPORT.md`](CONVEX_SUPPORT.md)
- convex MIP details:
  [`CONVEX_MIP_SUPPORT.md`](CONVEX_MIP_SUPPORT.md)
- viewer behavior:
  [`VISUALIZER_WORKFLOW.md`](VISUALIZER_WORKFLOW.md)

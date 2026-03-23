# Convex Support

This repo now includes a first-pass convex probe path alongside the original
linear path.

## What Is Implemented

- `probe_type=convex` in the current config/CLI flow
- convex ray generation in the active training/inference path
- convex target remapping for training
- convex sweep-fusion mapping for the visualizer
- convex probe overlay geometry in the viewer
- checkpoint-backed convex rendering in the visualizer

The shared implementation lives mainly in:

- [`src/ultranerf/probe_geometry.py`](../src/ultranerf/probe_geometry.py)
- [`src/ultranerf/nerf_utils.py`](../src/ultranerf/nerf_utils.py)
- [`src/ultranerf/visualization/transforms.py`](../src/ultranerf/visualization/transforms.py)

## Required Convex Parameters

The current convex path uses:

- `probe_type = convex`
- `convex_center_x`
- `convex_center_y`
- `convex_angle_deg`
- `convex_outer_radius_px`
- `convex_inner_radius_px`
- `convex_scale_x_mm`
- `convex_scale_y_mm`
- `convex_n_rays`
- `convex_n_samples`

Example config:

- [`configs/config_convex_example.txt`](../configs/config_convex_example.txt)

## Current Runtime Convention

- training/inference renders in the convex fan grid with shape:
  - `(convex_n_samples, convex_n_rays)`
- the visualizer remaps convex render outputs back into the original image
  layout for review
- sweep fusion uses the convex fan geometry instead of the linear rectangular
  scan-plane assumption

## Dataset Conversion

For tracked convex datasets that are not already in UltraNeRF format, use:

- [`convert_convex_dataset.py`](../convert_convex_dataset.py)

Expected source layout:

- `fan_info.xml`
- one sweep directory per acquisition
- image frames such as `*.png`
- one `tracking.csv` per sweep

The converter writes:

- `images.npy`
- `poses.npy`
- one converted sweep folder per source sweep
- a generated `multi_sweep_manifest.json`

Example:

```bash
python convert_convex_dataset.py \
  --source-root /workspace/data/abdominal_phantom \
  --output-root /workspace/data/abdominal_phantom_ultranerf
```

The current converter assumes the raw tracking translation is attached to the
raw image center and shifts it to the convex fan center expected by the current
UltraNeRF convex pose convention.

## Scope Limits

This is a selective port from `legacy/convex_probe/`.

Included:

- baseline convex geometry
- baseline convex rendering layout
- baseline convex viewer support

Not included yet:

- legacy convex MIP rendering
- legacy full-volume convex branches
- legacy cache/debug branches
- legacy concentric/radial sampling alternatives

## MIP Status

The config parser includes `render_mode`, but:

- `render_mode=default` is the supported path
- `render_mode=convex_mip` currently raises `NotImplementedError`

That is intentional until the dedicated MIP migration tickets are implemented.

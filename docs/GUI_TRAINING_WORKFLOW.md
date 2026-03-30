# GUI Training Workflow

The training workflow now has its own standalone entry point.

Launch it with:

```bash
python run_training_gui.py
```

The intended workflow is:

1. Launch `run_training_gui.py`.
2. Select a root folder containing compatible sweep subfolders with:
   - `images.npy`
   - `poses.npy`
3. Discover sweeps.
4. Choose:
   - one or more training sweeps
   - one validation sweep
5. Enter the probe geometry.
6. Open the preview and visually confirm the placement.
7. Choose a training scheme from the dropdown.
8. Start training.
9. After a successful run, click `Open Result in Viewer`.

What the GUI does internally:

- concatenates the selected sweeps into one prepared dataset
- sanitizes non-finite values
- clips large outliers using a sampled dataset-specific percentile cap
- rescales the prepared images back into the loader-friendly `0..255` range
- writes a `split.json` file for explicit train/validation indices
- writes a generated training config that references the selected scheme
- launches `run_ultranerf.py` in the background
- polls `progress.jsonl`
- updates the dialog when validation preview GIFs appear
- can open the finished result directly in the visualization viewer

Artifacts are written under:

- `logs/gui_training/<run_id>/`

Key files there:

- `generated_training_config.txt`
- `dataset/images.npy`
- `dataset/poses.npy`
- `dataset/split.json`
- `resolved_training_config.json`
- `progress.jsonl`
- `validation_preview/latest.gif`
- `stdout.log`

Current scope:

- the GUI launches `run_ultranerf.py` in a background subprocess
- sweep discovery expects `.npy` training datasets that already match the repo's standard format
- training stays in a background subprocess; the GUI monitors progress rather than running optimization inside napari
- the validation preview is an animated multi-pose summary rather than a single static image
- opening the finished result viewer closes the training dialog

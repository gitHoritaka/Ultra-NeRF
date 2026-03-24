# GUI Training Workflow

The multi-sweep viewer now includes a `Training` launcher on the left side.

The intended workflow is:

1. Open the training dialog.
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

What the GUI does internally:

- concatenates the selected sweeps into one prepared dataset
- writes a `split.json` file for explicit train/validation indices
- writes a generated training config that references the selected scheme
- launches `run_ultranerf.py` in the background
- polls `progress.jsonl`
- updates the dialog when validation previews appear

Artifacts are written under:

- `logs/gui_training/<run_id>/`

Key files there:

- `generated_training_config.txt`
- `dataset/images.npy`
- `dataset/poses.npy`
- `dataset/split.json`
- `resolved_training_config.json`
- `progress.jsonl`
- `validation_preview/latest.png`
- `stdout.log`

Current scope:

- the GUI launches baseline `run_ultranerf.py`
- sweep discovery expects `.npy` training datasets that already match the repo's standard format
- training stays in a background subprocess; the GUI monitors progress rather than running optimization inside napari

# Training Config

UltraNeRF training is now driven by two layers:

- a flat `configargparse` config passed to `run_ultranerf.py`
- an optional JSON `training_scheme` referenced from that config

For a practical terminal workflow, see [`TRAINING_CLI.md`](TRAINING_CLI.md).
For the standalone GUI workflow, see [`GUI_TRAINING_WORKFLOW.md`](GUI_TRAINING_WORKFLOW.md).

The flat config still carries dataset and probe/runtime details such as:

- `datadir`
- `split_file`
- `probe_type`
- probe geometry values
- `basedir`
- `expname`

The `training_scheme` file is where reusable recipe behavior lives:

- `n_iters`
- print / image / checkpoint cadence
- validation preview cadence
- loss terms and weights
- regularization toggles

Available schemes currently live under [`configs/training_schemes/`](../configs/training_schemes).

Example:

```ini
expname = gui_train_example
basedir = ./logs/gui_training
datadir = ./data/my_prepared_dataset
split_file = ./data/my_prepared_dataset/split.json
probe_type = linear
probe_width = 80
probe_depth = 140
training_scheme = ./configs/training_schemes/ssim_l2_balanced.json
```

At run start, `run_ultranerf.py` resolves the final config and writes:

- `args.txt`
- `config.txt` if a source config file was provided
- `resolved_training_config.json`

under the run directory.

Validation preview support:

- if `split_file` provides validation indices
- and `validation_preview_every > 0`

then the training loop writes:

- `progress.jsonl`
- `validation_preview/latest.gif`

for GUI monitoring.

GUI dataset preparation note:

- the GUI training path sanitizes non-finite values
- clips large outliers with a sampled per-dataset percentile cap
- rescales prepared images into the `0..255` range expected by `load_us.py`

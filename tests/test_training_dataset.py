import json
from pathlib import Path

import numpy as np

from ultranerf.probe_geometry import ProbeGeometry
from ultranerf.training_dataset import (
    build_preview_manifest,
    build_training_dataset_from_sweeps,
    discover_training_sweeps,
)


def write_sweep(path: Path, offset: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    images = np.full((2, 4, 5), offset, dtype=np.float32)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    poses[:, 0, 3] = offset
    np.save(path / "images.npy", images)
    np.save(path / "poses.npy", poses)


def test_discover_training_sweeps_reads_basic_metadata(tmp_path: Path) -> None:
    write_sweep(tmp_path / "a", 1.0)
    write_sweep(tmp_path / "b", 2.0)
    sweeps = discover_training_sweeps(tmp_path)
    assert tuple(sweep.sweep_id for sweep in sweeps) == ("a", "b")
    assert sweeps[0].image_shape == (4, 5)


def test_build_training_dataset_from_sweeps_concatenates_and_writes_split(tmp_path: Path) -> None:
    write_sweep(tmp_path / "train", 1.0)
    write_sweep(tmp_path / "val", 2.0)
    sweeps = discover_training_sweeps(tmp_path)
    artifacts = build_training_dataset_from_sweeps(
        sweeps,
        training_ids=("train",),
        validation_ids=("val",),
        output_dir=tmp_path / "out",
    )
    images = np.load(artifacts["images_path"])
    poses = np.load(artifacts["poses_path"])
    split = json.loads(Path(artifacts["split_path"]).read_text())
    assert images.shape == (4, 4, 5)
    assert poses.shape == (4, 4, 4)
    assert split["train_indices"] == [0, 1]
    assert split["validation_indices"] == [2, 3]


def test_build_preview_manifest_writes_selected_sweeps_and_geometry(tmp_path: Path) -> None:
    write_sweep(tmp_path / "a", 1.0)
    write_sweep(tmp_path / "b", 2.0)
    sweeps = discover_training_sweeps(tmp_path)
    geometry = ProbeGeometry(width_mm=20.0, depth_mm=30.0)
    manifest_path = build_preview_manifest(
        sweeps,
        selected_ids=("a",),
        probe_geometry=geometry,
        output_path=tmp_path / "preview.json",
    )
    payload = json.loads(manifest_path.read_text())
    assert payload["probe_geometry"]["width_mm"] == 20.0
    assert [entry["sweep_id"] for entry in payload["sweeps"]] == ["a"]

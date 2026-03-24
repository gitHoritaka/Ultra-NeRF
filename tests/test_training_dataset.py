import json
from pathlib import Path

import numpy as np

from ultranerf.probe_geometry import ProbeGeometry
from ultranerf.training_dataset import (
    build_preview_manifest,
    build_training_dataset_from_sweeps,
    discover_training_sweeps,
    estimate_training_clip_max,
    rescale_training_images_to_loader_range,
    sanitize_training_images,
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


def test_estimate_training_clip_max_uses_dataset_percentile() -> None:
    images = (
        np.asarray([[[0.0, 10.0, 20.0, 30.0, 1000.0]]], dtype=np.float32),
        np.asarray([[[5.0, 15.0, 25.0, 35.0, 45.0]]], dtype=np.float32),
    )
    clip_max = estimate_training_clip_max(images, percentile=90.0, max_samples_per_sweep=100)
    assert 30.0 <= clip_max <= 1000.0


def test_sanitize_training_images_replaces_nonfinite_and_clips() -> None:
    images = np.asarray(
        [[[np.nan, np.inf, -np.inf, -5.0, 42.0, 400.0]]],
        dtype=np.float32,
    )
    sanitized = sanitize_training_images(images, max_value=123.0)
    assert np.isfinite(sanitized).all()
    assert sanitized.min() == 0.0
    assert sanitized.max() == 123.0
    assert sanitized[0, 0, 4] == 42.0


def test_rescale_training_images_to_loader_range_maps_clip_ceiling_to_255() -> None:
    images = np.asarray([[[0.0, 21.0, 42.0]]], dtype=np.float32)
    scaled = rescale_training_images_to_loader_range(images, source_max_value=42.0)
    assert np.isclose(float(scaled.min()), 0.0)
    assert np.isclose(float(scaled.max()), 255.0)
    assert np.isclose(float(scaled[0, 0, 1]), 127.5)


def test_build_training_dataset_from_sweeps_sanitizes_invalid_values(tmp_path: Path) -> None:
    write_sweep(tmp_path / "train", 1.0)
    write_sweep(tmp_path / "val", 2.0)
    train_images = np.load(tmp_path / "train" / "images.npy")
    train_images[0, 0, 0] = np.nan
    train_images[0, 0, 1] = np.inf
    train_images[0, 0, 2] = -10.0
    train_images[0, 0, 3] = 999.0
    np.save(tmp_path / "train" / "images.npy", train_images)

    sweeps = discover_training_sweeps(tmp_path)
    artifacts = build_training_dataset_from_sweeps(
        sweeps,
        training_ids=("train",),
        validation_ids=("val",),
        output_dir=tmp_path / "out",
    )
    images = np.load(artifacts["images_path"])
    manifest = json.loads(Path(artifacts["manifest_path"]).read_text())
    assert np.isfinite(images).all()
    assert float(images.min()) >= 0.0
    assert float(images.max()) <= 255.0
    assert manifest["sanitization"]["nonfinite_values_replaced"] == 2
    assert manifest["sanitization"]["finite_values_clipped"] == 2
    assert manifest["sanitization"]["clip_max_percentile"] == 99.0
    assert manifest["sanitization"]["rescaled_to_loader_range"] is True

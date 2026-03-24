"""Helpers for discovering sweeps and preparing datasets for GUI-launched training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from ultranerf.probe_geometry import ProbeGeometry
from ultranerf.training_config import DatasetSplit, write_dataset_split


DEFAULT_HIGH_CLIP_PERCENTILE = 99.0
DEFAULT_MAX_SAMPLES_PER_SWEEP = 200_000


def estimate_training_clip_max(
    images: tuple[np.ndarray, ...],
    *,
    percentile: float = DEFAULT_HIGH_CLIP_PERCENTILE,
    max_samples_per_sweep: int = DEFAULT_MAX_SAMPLES_PER_SWEEP,
) -> float:
    """Estimate a stable per-dataset upper clip from finite image values."""
    if percentile <= 0.0 or percentile > 100.0:
        raise ValueError("percentile must be in the interval (0, 100]")
    samples: list[np.ndarray] = []
    for image_block in images:
        array = np.asarray(image_block, dtype=np.float32)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            continue
        finite = finite[finite >= 0.0]
        if finite.size == 0:
            continue
        if finite.size > max_samples_per_sweep:
            indices = np.linspace(0, finite.size - 1, num=max_samples_per_sweep, dtype=np.int64)
            finite = finite[indices]
        samples.append(finite.astype(np.float32, copy=False))
    if not samples:
        return 255.0
    merged = np.concatenate(samples, axis=0)
    clip_max = float(np.percentile(merged, percentile))
    if not np.isfinite(clip_max) or clip_max <= 0.0:
        return 255.0
    return clip_max


def sanitize_training_images(images: np.ndarray, *, min_value: float = 0.0, max_value: float = 255.0) -> np.ndarray:
    """Sanitize GUI-selected training images before writing the combined dataset.

    The sweep exports may contain NaN/inf values and extreme outliers. The GUI
    training path should clamp those into a stable image range rather than let
    them poison optimization immediately.
    """
    array = np.asarray(images, dtype=np.float32)
    sanitized = np.where(np.isfinite(array), array, float(min_value))
    sanitized = np.clip(sanitized, float(min_value), float(max_value))
    return sanitized.astype(np.float32, copy=False)


def rescale_training_images_to_loader_range(
    images: np.ndarray,
    *,
    source_max_value: float,
    target_max_value: float = 255.0,
) -> np.ndarray:
    """Rescale sanitized images into the range expected by ``load_us.py``."""
    array = np.asarray(images, dtype=np.float32)
    if source_max_value <= 0.0:
        return np.zeros_like(array, dtype=np.float32)
    scale = float(target_max_value) / float(source_max_value)
    return (array * scale).astype(np.float32, copy=False)


@dataclass(frozen=True)
class DiscoveredTrainingSweep:
    """One sweep directory that looks compatible with UltraNeRF training."""

    sweep_id: str
    dataset_dir: Path
    frame_count: int
    image_shape: tuple[int, int]
    dtype: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "dataset_dir": str(self.dataset_dir.resolve()),
            "frame_count": self.frame_count,
            "image_shape": list(self.image_shape),
            "dtype": self.dtype,
        }


def discover_training_sweeps(root_dir: str | Path) -> tuple[DiscoveredTrainingSweep, ...]:
    """Find compatible `images.npy`/`poses.npy` sweep directories under a root."""
    root = Path(root_dir)
    discovered: list[DiscoveredTrainingSweep] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        images_path = child / "images.npy"
        poses_path = child / "poses.npy"
        if not images_path.exists() or not poses_path.exists():
            continue
        images = np.load(images_path, mmap_mode="r")
        poses = np.load(poses_path, mmap_mode="r")
        if images.ndim != 3 or poses.ndim != 3 or images.shape[0] != poses.shape[0]:
            continue
        discovered.append(
            DiscoveredTrainingSweep(
                sweep_id=child.name,
                dataset_dir=child,
                frame_count=int(images.shape[0]),
                image_shape=(int(images.shape[1]), int(images.shape[2])),
                dtype=str(images.dtype),
            )
        )
    return tuple(discovered)


def validate_sweep_selection(
    sweeps: tuple[DiscoveredTrainingSweep, ...],
    *,
    training_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
) -> None:
    """Validate a GUI-selected train/validation split."""
    if not training_ids:
        raise ValueError("At least one training sweep must be selected")
    if not validation_ids:
        raise ValueError("At least one validation sweep must be selected")
    known_ids = {sweep.sweep_id for sweep in sweeps}
    for sweep_id in training_ids + validation_ids:
        if sweep_id not in known_ids:
            raise ValueError(f"Unknown sweep selected: {sweep_id}")
    selected_shapes = {
        sweep.image_shape
        for sweep in sweeps
        if sweep.sweep_id in set(training_ids + validation_ids)
    }
    if len(selected_shapes) != 1:
        raise ValueError("Selected training and validation sweeps must share the same image shape")


def build_training_dataset_from_sweeps(
    sweeps: tuple[DiscoveredTrainingSweep, ...],
    *,
    training_ids: tuple[str, ...],
    validation_ids: tuple[str, ...],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Concatenate selected sweeps into one dataset plus explicit split metadata."""
    validate_sweep_selection(sweeps, training_ids=training_ids, validation_ids=validation_ids)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    sweep_lookup = {sweep.sweep_id: sweep for sweep in sweeps}
    selected_ids = tuple(dict.fromkeys(training_ids + validation_ids))
    image_chunks: list[np.ndarray] = []
    pose_chunks: list[np.ndarray] = []
    train_indices: list[int] = []
    validation_indices: list[int] = []
    offset = 0
    sweep_offsets: dict[str, dict[str, Any]] = {}
    sanitized_nonfinite_count = 0
    clipped_value_count = 0
    raw_image_blocks: list[np.ndarray] = []

    for sweep_id in selected_ids:
        sweep = sweep_lookup[sweep_id]
        raw_image_blocks.append(np.load(sweep.dataset_dir / "images.npy").astype(np.float32))

    clip_max = estimate_training_clip_max(tuple(raw_image_blocks))

    for sweep_id, raw_images in zip(selected_ids, raw_image_blocks):
        sweep = sweep_lookup[sweep_id]
        sanitized_nonfinite_count += int(np.size(raw_images) - np.count_nonzero(np.isfinite(raw_images)))
        clipped_value_count += int(np.count_nonzero(np.isfinite(raw_images) & ((raw_images < 0.0) | (raw_images > clip_max))))
        images = sanitize_training_images(raw_images, max_value=clip_max)
        images = rescale_training_images_to_loader_range(images, source_max_value=clip_max)
        poses = np.load(sweep.dataset_dir / "poses.npy").astype(np.float32)
        frame_count = int(images.shape[0])
        image_chunks.append(images)
        pose_chunks.append(poses)
        local_indices = list(range(offset, offset + frame_count))
        if sweep_id in training_ids:
            train_indices.extend(local_indices)
        if sweep_id in validation_ids:
            validation_indices.extend(local_indices)
        sweep_offsets[sweep_id] = {
            "offset": offset,
            "frame_count": frame_count,
            "dataset_dir": str(sweep.dataset_dir.resolve()),
        }
        offset += frame_count

    combined_images = np.concatenate(image_chunks, axis=0)
    combined_poses = np.concatenate(pose_chunks, axis=0)
    np.save(output_root / "images.npy", combined_images)
    np.save(output_root / "poses.npy", combined_poses)

    split = DatasetSplit(
        train_indices=tuple(train_indices),
        validation_indices=tuple(validation_indices),
        metadata={"selected_training_sweeps": list(training_ids), "selected_validation_sweeps": list(validation_ids)},
    )
    split_path = write_dataset_split(output_root / "split.json", split)

    manifest_payload = {
        "frame_count": int(combined_images.shape[0]),
        "image_shape": list(combined_images.shape[1:]),
        "selected_training_sweeps": list(training_ids),
        "selected_validation_sweeps": list(validation_ids),
        "sweep_offsets": sweep_offsets,
        "sanitization": {
            "clip_min": 0.0,
            "clip_max": clip_max,
            "clip_max_percentile": DEFAULT_HIGH_CLIP_PERCENTILE,
            "rescaled_to_loader_range": True,
            "loader_target_max": 255.0,
            "nonfinite_values_replaced": sanitized_nonfinite_count,
            "finite_values_clipped": clipped_value_count,
        },
    }
    manifest_path = output_root / "training_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2))
    return {
        "dataset_dir": output_root,
        "images_path": output_root / "images.npy",
        "poses_path": output_root / "poses.npy",
        "split_path": split_path,
        "manifest_path": manifest_path,
    }


def build_preview_manifest(
    sweeps: tuple[DiscoveredTrainingSweep, ...],
    *,
    selected_ids: tuple[str, ...],
    probe_geometry: ProbeGeometry,
    output_path: str | Path,
) -> Path:
    """Write a temporary multi-sweep manifest for training preview."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "probe_geometry": {
            "width_mm": float(probe_geometry.width_mm),
            "depth_mm": float(probe_geometry.depth_mm),
            "probe_type": probe_geometry.probe_type,
            "convex_center_x": probe_geometry.convex_center_x,
            "convex_center_y": probe_geometry.convex_center_y,
            "convex_angle_deg": probe_geometry.convex_angle_deg,
            "convex_outer_radius_px": probe_geometry.convex_outer_radius_px,
            "convex_inner_radius_px": probe_geometry.convex_inner_radius_px,
            "convex_scale_x_mm": probe_geometry.convex_scale_x_mm,
            "convex_scale_y_mm": probe_geometry.convex_scale_y_mm,
            "convex_n_rays": probe_geometry.convex_n_rays,
            "convex_n_samples": probe_geometry.convex_n_samples,
        },
        "active_sweep_id": selected_ids[0] if selected_ids else None,
        "comparison_policy": "all_enabled",
        "metadata": {"source": "gui_training_preview"},
        "sweeps": [
            {
                "sweep_id": sweep.sweep_id,
                "dataset_dir": str(sweep.dataset_dir.resolve()),
                "display_name": sweep.sweep_id,
            }
            for sweep in sweeps
            if sweep.sweep_id in selected_ids
        ],
    }
    output.write_text(json.dumps(payload, indent=2))
    return output

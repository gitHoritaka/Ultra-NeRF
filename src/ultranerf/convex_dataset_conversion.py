"""Utilities for converting tracked convex ultrasound sweeps to UltraNeRF format."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ConvexFanMetadata:
    """Convex fan geometry parsed from an XML description."""

    center_x_px: float
    center_y_px: float
    inner_radius_px: float
    outer_radius_px: float
    opening_angle_deg: float
    image_width_px: int
    image_height_px: int
    spacing_x_mm: float
    spacing_y_mm: float

    @property
    def inner_radius_mm(self) -> float:
        return float(self.inner_radius_px) * float(self.spacing_y_mm)

    @property
    def outer_radius_mm(self) -> float:
        return float(self.outer_radius_px) * float(self.spacing_y_mm)

    @property
    def depth_mm(self) -> float:
        return self.outer_radius_mm - self.inner_radius_mm

    @property
    def width_mm(self) -> float:
        half_angle = np.deg2rad(float(self.opening_angle_deg) * 0.5)
        return float(2.0 * self.outer_radius_mm * np.sin(half_angle))

    @property
    def image_center_px(self) -> tuple[float, float]:
        return ((float(self.image_width_px) - 1.0) * 0.5, (float(self.image_height_px) - 1.0) * 0.5)


def _read_param_map(xml_path: Path) -> dict[str, str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    params: dict[str, str] = {}
    for node in root.findall(".//param"):
        name = node.attrib.get("name")
        if name is None or node.text is None:
            continue
        params[name] = node.text.strip()
    return params


def parse_fan_info_xml(xml_path: str | Path) -> ConvexFanMetadata:
    """Parse XML fan metadata exported with the convex dataset."""
    path = Path(xml_path)
    params = _read_param_map(path)
    offset_x, offset_y = (float(v) for v in params["offset"].split())
    spacing_x, spacing_y, _spacing_z = (float(v) for v in params["spacing"].split())
    # The source XML stores openingAngle as the half-angle of the fan.
    opening_half_angle_deg = float(params["openingAngle"])
    return ConvexFanMetadata(
        center_x_px=offset_x,
        center_y_px=offset_y,
        inner_radius_px=float(params["shortRadius"]),
        outer_radius_px=float(params["longRadius"]),
        opening_angle_deg=opening_half_angle_deg * 2.0,
        image_width_px=int(params["width"]),
        image_height_px=int(params["height"]),
        spacing_x_mm=spacing_x,
        spacing_y_mm=spacing_y,
    )


def parse_tracking_pose_row(values: Iterable[str] | np.ndarray) -> np.ndarray:
    """Parse one tracking row into a conventional 4x4 probe-to-world matrix in mm.

    The abdominal phantom tracking stores 18 tab-separated values per row.
    The first 16 values form a flattened 4x4 matrix in column-major order.
    The last 2 values are ignored.
    """
    raw = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if raw.size < 16:
        raise ValueError("tracking row must contain at least 16 values")
    pose = raw[:16].reshape(4, 4).T.astype(np.float32)
    return pose


def fan_center_offset_mm_from_image_center(metadata: ConvexFanMetadata) -> np.ndarray:
    """Return the fixed local offset from image-center tracking origin to fan center.

    The current UltraNeRF convex convention expects the probe pose translation to
    be located at the fan center. Some tracked datasets provide the pose at the
    center of the raw image instead. In that case, a constant local translation
    must be applied.
    """
    image_center_x, image_center_y = metadata.image_center_px
    return np.array(
        [
            (metadata.center_x_px - image_center_x) * metadata.spacing_x_mm,
            (metadata.center_y_px - image_center_y) * metadata.spacing_y_mm,
            0.0,
        ],
        dtype=np.float32,
    )


def apply_local_translation(pose_tracking_origin_mm: np.ndarray, offset_mm: np.ndarray) -> np.ndarray:
    """Shift a probe pose by a fixed local-frame translation."""
    pose = np.asarray(pose_tracking_origin_mm, dtype=np.float32).copy()
    offset = np.eye(4, dtype=np.float32)
    offset[:3, 3] = np.asarray(offset_mm, dtype=np.float32).reshape(3)
    return (pose @ offset).astype(np.float32)


def load_png_stack(image_paths: Iterable[Path]) -> np.ndarray:
    """Load a stack of grayscale PNG images into an array of shape [N, H, W]."""
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(np.array(image.convert("L"), dtype=np.uint8))
    if not images:
        raise ValueError("no image files were found")
    stack = np.stack(images, axis=0)
    return stack.astype(np.uint8)


def load_tracking_poses(tracking_csv_path: str | Path, *, tracking_origin: str, metadata: ConvexFanMetadata) -> np.ndarray:
    """Load tracking poses and convert them to UltraNeRF's convex origin convention."""
    path = Path(tracking_csv_path)
    rows = [line.strip().split("\t") for line in path.read_text().splitlines() if line.strip()]
    poses = np.stack([parse_tracking_pose_row(row) for row in rows], axis=0).astype(np.float32)
    if tracking_origin == "image_center":
        offset_mm = fan_center_offset_mm_from_image_center(metadata)
        poses = np.stack([apply_local_translation(pose, offset_mm) for pose in poses], axis=0)
    elif tracking_origin != "fan_center":
        raise ValueError(f"unsupported tracking_origin: {tracking_origin}")
    return poses.astype(np.float32)


def convert_convex_sweep(
    *,
    source_sweep_dir: str | Path,
    output_sweep_dir: str | Path,
    metadata: ConvexFanMetadata,
    tracking_origin: str = "image_center",
    image_glob: str = "*.png",
    overwrite: bool = False,
) -> dict[str, object]:
    """Convert one raw convex sweep directory into UltraNeRF dataset files."""
    source_dir = Path(source_sweep_dir)
    output_dir = Path(output_sweep_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_path = output_dir / "images.npy"
    poses_path = output_dir / "poses.npy"
    if not overwrite and (images_path.exists() or poses_path.exists()):
        raise FileExistsError(f"refusing to overwrite existing outputs in {output_dir}")

    image_paths = sorted(source_dir.glob(image_glob))
    images = load_png_stack(image_paths)
    if images.shape[1] != metadata.image_height_px or images.shape[2] != metadata.image_width_px:
        raise ValueError(
            f"image stack shape {images.shape[1:]} does not match fan XML "
            f"shape {(metadata.image_height_px, metadata.image_width_px)}"
        )

    poses = load_tracking_poses(source_dir / "tracking.csv", tracking_origin=tracking_origin, metadata=metadata)
    if poses.shape[0] != images.shape[0]:
        raise ValueError(f"frame/pose count mismatch: {images.shape[0]} images vs {poses.shape[0]} poses")

    np.save(images_path, images)
    np.save(poses_path, poses.astype(np.float32))

    summary = {
        "source_sweep_dir": str(source_dir.resolve()),
        "output_sweep_dir": str(output_dir.resolve()),
        "num_frames": int(images.shape[0]),
        "image_shape": [int(images.shape[1]), int(images.shape[2])],
        "tracking_origin": tracking_origin,
    }
    (output_dir / "conversion_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_multi_sweep_manifest(
    *,
    output_root: str | Path,
    sweep_names: Iterable[str],
    metadata: ConvexFanMetadata,
    convex_n_rays: int,
    convex_n_samples: int,
    manifest_name: str = "multi_sweep_manifest.json",
) -> Path:
    """Write a multi-sweep visualization manifest for converted convex data."""
    root = Path(output_root)
    manifest = {
        "probe_geometry": {
            "probe_type": "convex",
            "width_mm": metadata.width_mm,
            "depth_mm": metadata.depth_mm,
            "convex_center_x": metadata.center_x_px,
            "convex_center_y": metadata.center_y_px,
            "convex_angle_deg": metadata.opening_angle_deg,
            "convex_outer_radius_px": metadata.outer_radius_px,
            "convex_inner_radius_px": metadata.inner_radius_px,
            "convex_scale_x_mm": metadata.spacing_x_mm,
            "convex_scale_y_mm": metadata.spacing_y_mm,
            "convex_n_rays": int(convex_n_rays),
            "convex_n_samples": int(convex_n_samples),
        },
        "active_sweep_id": next(iter(sweep_names), None),
        "comparison_policy": "all_enabled",
        "metadata": {
            "source_dataset": "converted_convex_tracking_dataset",
            "fan_info": asdict(metadata),
        },
        "sweeps": [
            {
                "sweep_id": str(name),
                "display_name": str(name),
                "dataset_dir": str(name),
                "enabled": True,
                "alignment_source": "tracked_convex_dataset",
            }
            for name in sweep_names
        ],
    }
    path = root / manifest_name
    path.write_text(json.dumps(manifest, indent=2))
    return path


def convert_convex_multi_sweep_dataset(
    *,
    source_root: str | Path,
    output_root: str | Path,
    fan_info_xml: str | Path | None = None,
    tracking_origin: str = "image_center",
    sweep_glob: str = "sweep_*",
    image_glob: str = "*.png",
    convex_n_rays: int | None = None,
    convex_n_samples: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Convert a raw convex multi-sweep dataset into UltraNeRF format."""
    source = Path(source_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)

    metadata = parse_fan_info_xml(fan_info_xml or (source / "fan_info.xml"))
    sweep_dirs = tuple(sorted(p for p in source.glob(sweep_glob) if p.is_dir()))
    if not sweep_dirs:
        raise ValueError(f"no sweep directories matched {sweep_glob!r} under {source}")

    summaries = []
    for sweep_dir in sweep_dirs:
        summary = convert_convex_sweep(
            source_sweep_dir=sweep_dir,
            output_sweep_dir=output / sweep_dir.name,
            metadata=metadata,
            tracking_origin=tracking_origin,
            image_glob=image_glob,
            overwrite=overwrite,
        )
        summaries.append(summary)

    manifest_path = build_multi_sweep_manifest(
        output_root=output,
        sweep_names=[sweep_dir.name for sweep_dir in sweep_dirs],
        metadata=metadata,
        convex_n_rays=int(convex_n_rays if convex_n_rays is not None else metadata.image_width_px),
        convex_n_samples=int(convex_n_samples if convex_n_samples is not None else metadata.image_height_px),
    )

    conversion_summary = {
        "source_root": str(source.resolve()),
        "output_root": str(output.resolve()),
        "fan_info_xml": str(Path(fan_info_xml or (source / "fan_info.xml")).resolve()),
        "tracking_origin": tracking_origin,
        "manifest_path": str(manifest_path.resolve()),
        "num_sweeps": len(summaries),
        "sweeps": summaries,
    }
    (output / "conversion_summary.json").write_text(json.dumps(conversion_summary, indent=2))
    return conversion_summary

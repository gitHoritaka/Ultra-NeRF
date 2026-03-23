from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ultranerf.convex_dataset_conversion import (
    apply_local_translation,
    build_multi_sweep_manifest,
    convert_convex_multi_sweep_dataset,
    fan_center_offset_mm_from_image_center,
    parse_fan_info_xml,
    parse_tracking_pose_row,
)


def write_fan_xml(path: Path) -> None:
    path.write_text(
        """<root>
  <param name="offset">5 2</param>
  <param name="shortRadius">2</param>
  <param name="longRadius">10</param>
  <param name="openingAngle">30</param>
  <param name="width">11</param>
  <param name="height">7</param>
  <param name="spacing">0.5 0.25 1</param>
</root>
"""
    )


def write_sweep(path: Path, name: str, num_frames: int = 2) -> None:
    sweep = path / name
    sweep.mkdir(parents=True)
    for idx in range(num_frames):
        image = np.full((7, 11), idx + 1, dtype=np.uint8)
        Image.fromarray(image).save(sweep / f"frame_{idx:04d}.png")
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    flattened = pose.T.reshape(-1)
    rows = []
    for idx in range(num_frames):
        with_offset = flattened.copy()
        with_offset[12] += idx
        rows.append("\t".join(f"{value:.6f}" for value in np.concatenate([with_offset, [123.0 + idx, 1.0]])))
    (sweep / "tracking.csv").write_text("\n".join(rows) + "\n")


def test_parse_tracking_pose_row_transposes_column_major_flattened_input() -> None:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    row = pose.T.reshape(-1)
    parsed = parse_tracking_pose_row(row)
    np.testing.assert_allclose(parsed, pose)


def test_fan_center_offset_uses_image_center_in_pixel_coordinates(tmp_path: Path) -> None:
    xml_path = tmp_path / "fan_info.xml"
    write_fan_xml(xml_path)
    metadata = parse_fan_info_xml(xml_path)
    assert metadata.opening_angle_deg == 60.0
    offset = fan_center_offset_mm_from_image_center(metadata)
    np.testing.assert_allclose(offset, np.array([0.0, -0.25, 0.0], dtype=np.float32))


def test_apply_local_translation_shifts_pose_translation() -> None:
    pose = np.eye(4, dtype=np.float32)
    shifted = apply_local_translation(pose, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(shifted[:3, 3], np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_convert_convex_multi_sweep_dataset_writes_images_poses_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    write_fan_xml(source / "fan_info.xml")
    write_sweep(source, "sweep_1")
    write_sweep(source, "sweep_2")

    output = tmp_path / "converted"
    summary = convert_convex_multi_sweep_dataset(source_root=source, output_root=output)

    images = np.load(output / "sweep_1" / "images.npy")
    poses = np.load(output / "sweep_1" / "poses.npy")
    manifest = json.loads((output / "multi_sweep_manifest.json").read_text())

    assert images.shape == (2, 7, 11)
    assert poses.shape == (2, 4, 4)
    assert manifest["probe_geometry"]["probe_type"] == "convex"
    assert manifest["probe_geometry"]["convex_angle_deg"] == 60.0
    assert manifest["sweeps"][0]["dataset_dir"] == "sweep_1"
    assert summary["num_sweeps"] == 2

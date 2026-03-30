"""Geometric representation of the probe and scan plane for 3D viewers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultranerf.visualization.transforms import ProbeGeometry, pose_to_axes, probe_local_to_world, probe_plane_corners


@dataclass(frozen=True)
class ProbeRepresentation:
    """Probe overlay geometry in world-space millimeters."""

    origin_mm: np.ndarray
    axes_endpoints_mm: dict[str, np.ndarray]
    scan_plane_corners_mm: np.ndarray
    beam_line_mm: np.ndarray
    probe_face_line_mm: np.ndarray


def _convex_scan_plane_polygon_local(
    geometry: ProbeGeometry,
    *,
    arc_samples: int = 33,
) -> np.ndarray:
    """Return a sampled convex sector polygon in probe-local coordinates."""
    half_angle = np.deg2rad(float(geometry.convex_angle_deg)) * 0.5
    inner = float(geometry.convex_inner_radius_mm)
    outer = float(geometry.convex_outer_radius_mm)
    n = max(int(arc_samples), 3)
    angles = np.linspace(-half_angle, half_angle, n, dtype=np.float32)

    outer_arc = np.stack(
        [
            np.sin(angles) * outer,
            np.cos(angles) * outer,
            np.zeros_like(angles),
        ],
        axis=-1,
    ).astype(np.float32)
    inner_arc = np.stack(
        [
            np.sin(angles[::-1]) * inner,
            np.cos(angles[::-1]) * inner,
            np.zeros_like(angles),
        ],
        axis=-1,
    ).astype(np.float32)
    return np.concatenate([outer_arc, inner_arc], axis=0).astype(np.float32)


def build_probe_representation(
    pose_probe_to_world: np.ndarray,
    geometry: ProbeGeometry,
    *,
    axis_length_mm: float | None = None,
) -> ProbeRepresentation:
    """Build a simple world-space representation of the probe and scan plane."""
    origin, x_axis, y_axis, z_axis = pose_to_axes(pose_probe_to_world)
    axis_length = float(axis_length_mm if axis_length_mm is not None else max(geometry.width_mm, geometry.depth_mm) * 0.25)
    display_origin = origin.copy()
    if geometry.is_convex:
        display_origin = origin + y_axis * float(geometry.convex_inner_radius_mm)
        scan_plane = probe_local_to_world(
            _convex_scan_plane_polygon_local(geometry),
            pose_probe_to_world,
        )
        half_angle = np.deg2rad(float(geometry.convex_angle_deg)) * 0.5
        beam_local = np.array(
            [
                [0.0, geometry.convex_inner_radius_mm, 0.0],
                [0.0, geometry.convex_outer_radius_mm, 0.0],
            ],
            dtype=np.float32,
        )
        beam_line = probe_local_to_world(beam_local, pose_probe_to_world)
        probe_face_line = probe_local_to_world(
            np.array(
                [
                    [
                        np.sin(-half_angle) * geometry.convex_inner_radius_mm,
                        np.cos(-half_angle) * geometry.convex_inner_radius_mm,
                        0.0,
                    ],
                    [
                        np.sin(half_angle) * geometry.convex_inner_radius_mm,
                        np.cos(half_angle) * geometry.convex_inner_radius_mm,
                        0.0,
                    ],
                ],
                dtype=np.float32,
            ),
            pose_probe_to_world,
        )
    else:
        scan_plane = probe_plane_corners(pose_probe_to_world, geometry)
        beam_line = probe_local_to_world(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, geometry.depth_mm, 0.0],
                ],
                dtype=np.float32,
            ),
            pose_probe_to_world,
        )
        probe_face_line = probe_local_to_world(
            np.array(
                [
                    [-geometry.width_mm / 2.0, 0.0, 0.0],
                    [geometry.width_mm / 2.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            pose_probe_to_world,
        )
    axes_endpoints = {
        "x": display_origin + x_axis * axis_length,
        "y": display_origin + y_axis * axis_length,
        "z": display_origin + z_axis * axis_length,
    }
    return ProbeRepresentation(
        origin_mm=display_origin.astype(np.float32),
        axes_endpoints_mm={k: v.astype(np.float32) for k, v in axes_endpoints.items()},
        scan_plane_corners_mm=scan_plane.astype(np.float32),
        beam_line_mm=beam_line.astype(np.float32),
        probe_face_line_mm=probe_face_line.astype(np.float32),
    )

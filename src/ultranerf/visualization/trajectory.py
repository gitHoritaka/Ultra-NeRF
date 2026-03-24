"""Trajectory overlay helpers for tracked probe poses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ultranerf.probe_geometry import ProbeGeometry
from ultranerf.visualization.transforms import ensure_pose_matrix, pose_to_axes


@dataclass(frozen=True)
class TrajectoryOverlay:
    """Geometry needed to display a tracked trajectory in 3D."""

    centers_mm: np.ndarray
    polyline_mm: np.ndarray
    axis_origins_mm: np.ndarray
    axis_endpoints_mm: np.ndarray
    axis_labels: tuple[str, ...]


def trajectory_centers_from_poses(poses_probe_to_world: np.ndarray) -> np.ndarray:
    """Extract probe centers from a batch of poses."""
    poses = np.asarray(poses_probe_to_world, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[0] == 0:
        raise ValueError("poses_probe_to_world must have shape (N, 3, 4) or (N, 4, 4)")
    centers = [ensure_pose_matrix(pose)[:3, 3] for pose in poses]
    return np.asarray(centers, dtype=np.float32)


def trajectory_display_centers_from_poses(
    poses_probe_to_world: np.ndarray,
    *,
    probe_geometry: ProbeGeometry | None = None,
) -> np.ndarray:
    """Extract displayed trajectory centers from a batch of poses.

    For linear probes, the displayed trajectory matches the pose origin. For
    convex probes, the displayed trajectory is shifted to the midpoint of the
    inner arc so the path matches the visible imaging origin rather than the fan
    center.
    """
    poses = np.asarray(poses_probe_to_world, dtype=np.float32)
    centers = trajectory_centers_from_poses(poses)
    if probe_geometry is None or not probe_geometry.is_convex:
        return centers

    shifted = []
    inner_radius_mm = float(probe_geometry.convex_inner_radius_mm)
    for idx, pose in enumerate(poses):
        origin, _x_axis, y_axis, _z_axis = pose_to_axes(pose)
        shifted.append(origin + y_axis * inner_radius_mm)
    return np.asarray(shifted, dtype=np.float32)


def build_trajectory_overlay(
    poses_probe_to_world: np.ndarray,
    *,
    probe_geometry: ProbeGeometry | None = None,
    axis_stride: int = 10,
    axis_length_mm: float = 10.0,
) -> TrajectoryOverlay:
    """Build polyline and optional sampled axis geometry for the trajectory."""
    poses = np.asarray(poses_probe_to_world, dtype=np.float32)
    centers = trajectory_display_centers_from_poses(poses, probe_geometry=probe_geometry)

    sampled_origins = []
    sampled_endpoints = []
    sampled_labels = []
    for idx in range(0, poses.shape[0], max(axis_stride, 1)):
        origin, x_axis, y_axis, z_axis = pose_to_axes(poses[idx])
        if probe_geometry is not None and probe_geometry.is_convex:
            origin = origin + y_axis * float(probe_geometry.convex_inner_radius_mm)
        for label, axis in (("x", x_axis), ("y", y_axis), ("z", z_axis)):
            sampled_origins.append(origin)
            sampled_endpoints.append(origin + axis * axis_length_mm)
            sampled_labels.append(label)

    return TrajectoryOverlay(
        centers_mm=centers,
        polyline_mm=centers.copy(),
        axis_origins_mm=np.asarray(sampled_origins, dtype=np.float32),
        axis_endpoints_mm=np.asarray(sampled_endpoints, dtype=np.float32),
        axis_labels=tuple(sampled_labels),
    )


def nearest_trajectory_index(query_point_mm: np.ndarray, centers_mm: np.ndarray) -> int:
    """Return the nearest trajectory center index to a query point."""
    query = np.asarray(query_point_mm, dtype=np.float32).reshape(3)
    centers = np.asarray(centers_mm, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
        raise ValueError("centers_mm must have shape (N, 3)")
    distances = np.linalg.norm(centers - query[None, :], axis=1)
    return int(np.argmin(distances))

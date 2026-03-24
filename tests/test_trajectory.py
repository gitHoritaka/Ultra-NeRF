import numpy as np

from ultranerf.probe_geometry import ProbeGeometry
from ultranerf.visualization.trajectory import (
    build_trajectory_overlay,
    nearest_trajectory_index,
    trajectory_centers_from_poses,
    trajectory_display_centers_from_poses,
)


def translation_pose(tx: float, ty: float, tz: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return pose


def test_trajectory_centers_from_poses_extracts_translations():
    poses = np.stack(
        [
            translation_pose(0.0, 0.0, 0.0),
            translation_pose(1.0, 2.0, 3.0),
            translation_pose(4.0, 5.0, 6.0),
        ],
        axis=0,
    )

    centers = trajectory_centers_from_poses(poses)

    assert np.allclose(
        centers,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        ),
    )


def test_build_trajectory_overlay_creates_polyline_and_sampled_axes():
    poses = np.stack(
        [
            translation_pose(0.0, 0.0, 0.0),
            translation_pose(10.0, 0.0, 0.0),
            translation_pose(20.0, 0.0, 0.0),
            translation_pose(30.0, 0.0, 0.0),
        ],
        axis=0,
    )

    overlay = build_trajectory_overlay(poses, axis_stride=2, axis_length_mm=5.0)

    assert overlay.centers_mm.shape == (4, 3)
    assert np.allclose(overlay.polyline_mm, overlay.centers_mm)
    assert overlay.axis_origins_mm.shape == (6, 3)
    assert overlay.axis_endpoints_mm.shape == (6, 3)
    assert overlay.axis_labels == ("x", "y", "z", "x", "y", "z")
    assert np.allclose(overlay.axis_endpoints_mm[0], np.array([5.0, 0.0, 0.0], dtype=np.float32))


def test_nearest_trajectory_index_returns_closest_center():
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    index = nearest_trajectory_index(np.array([12.0, 1.0, 0.0], dtype=np.float32), centers)

    assert index == 1


def test_convex_trajectory_display_centers_use_inner_arc_midpoint():
    geometry = ProbeGeometry(
        width_mm=50.0,
        depth_mm=100.0,
        probe_type="convex",
        convex_center_x=0.0,
        convex_center_y=0.0,
        convex_angle_deg=70.0,
        convex_outer_radius_px=300.0,
        convex_inner_radius_px=20.0,
        convex_scale_x_mm=1.0,
        convex_scale_y_mm=2.0,
        convex_n_rays=32,
        convex_n_samples=64,
    )
    poses = np.stack(
        [
            translation_pose(1.0, 2.0, 3.0),
            translation_pose(4.0, 5.0, 6.0),
        ],
        axis=0,
    )

    display_centers = trajectory_display_centers_from_poses(poses, probe_geometry=geometry)
    overlay = build_trajectory_overlay(poses, probe_geometry=geometry, axis_stride=1, axis_length_mm=5.0)

    expected = np.array(
        [
            [1.0, 42.0, 3.0],
            [4.0, 45.0, 6.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(display_centers, expected)
    assert np.allclose(overlay.centers_mm, expected)
    assert np.allclose(overlay.axis_origins_mm[0], expected[0])

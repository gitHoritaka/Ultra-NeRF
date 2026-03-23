import numpy as np

from ultranerf.probe_geometry import ProbeGeometry
from ultranerf.visualization.probe_representation import build_probe_representation


def translation_pose(tx: float, ty: float, tz: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return pose


def test_probe_representation_matches_identity_pose_geometry():
    geometry = ProbeGeometry(width_mm=80.0, depth_mm=140.0)
    rep = build_probe_representation(np.eye(4, dtype=np.float32), geometry, axis_length_mm=10.0)

    assert np.allclose(rep.origin_mm, np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(rep.axes_endpoints_mm["x"], np.array([10.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(rep.axes_endpoints_mm["y"], np.array([0.0, 10.0, 0.0], dtype=np.float32))
    assert np.allclose(rep.axes_endpoints_mm["z"], np.array([0.0, 0.0, 10.0], dtype=np.float32))
    assert np.allclose(rep.beam_line_mm[1], np.array([0.0, 140.0, 0.0], dtype=np.float32))
    assert np.allclose(rep.probe_face_line_mm[0], np.array([-40.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(rep.probe_face_line_mm[1], np.array([40.0, 0.0, 0.0], dtype=np.float32))


def test_probe_representation_respects_pose_translation():
    geometry = ProbeGeometry(width_mm=20.0, depth_mm=30.0)
    rep = build_probe_representation(translation_pose(5.0, 6.0, 7.0), geometry, axis_length_mm=5.0)

    assert np.allclose(rep.origin_mm, np.array([5.0, 6.0, 7.0], dtype=np.float32))
    assert np.allclose(rep.axes_endpoints_mm["x"], np.array([10.0, 6.0, 7.0], dtype=np.float32))
    assert np.allclose(rep.beam_line_mm[1], np.array([5.0, 36.0, 7.0], dtype=np.float32))


def test_probe_representation_scan_plane_matches_expected_rectangle():
    geometry = ProbeGeometry(width_mm=10.0, depth_mm=20.0)
    rep = build_probe_representation(np.eye(4, dtype=np.float32), geometry)

    expected = np.array(
        [
            [-5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [5.0, 20.0, 0.0],
            [-5.0, 20.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(rep.scan_plane_corners_mm, expected)


def test_probe_representation_supports_convex_probe_geometry():
    geometry = ProbeGeometry(
        width_mm=80.0,
        depth_mm=140.0,
        probe_type="convex",
        convex_center_x=0.0,
        convex_center_y=0.0,
        convex_angle_deg=60.0,
        convex_outer_radius_px=40.0,
        convex_inner_radius_px=10.0,
        convex_scale_x_mm=1.0,
        convex_scale_y_mm=1.0,
        convex_n_rays=5,
        convex_n_samples=4,
    )
    rep = build_probe_representation(np.eye(4, dtype=np.float32), geometry, axis_length_mm=10.0)

    assert rep.scan_plane_corners_mm.shape == (4, 3)
    assert np.isclose(np.linalg.norm(rep.beam_line_mm[0, :2]), geometry.convex_inner_radius_mm, atol=1e-5)
    assert np.isclose(np.linalg.norm(rep.beam_line_mm[1, :2]), geometry.convex_outer_radius_mm, atol=1e-5)

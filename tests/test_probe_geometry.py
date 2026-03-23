from types import SimpleNamespace

import numpy as np

from ultranerf.probe_geometry import (
    ProbeGeometry,
    build_probe_geometry_from_args,
    convex_ray_grid_local_mm,
    convex_sampling_points_px,
    convex_valid_pixel_mask,
    remap_convex_grid_to_image,
    remap_image_to_convex_grid,
)


def make_convex_geometry() -> ProbeGeometry:
    return ProbeGeometry(
        width_mm=80.0,
        depth_mm=140.0,
        probe_type="convex",
        convex_center_x=50.0,
        convex_center_y=10.0,
        convex_angle_deg=60.0,
        convex_outer_radius_px=40.0,
        convex_inner_radius_px=10.0,
        convex_scale_x_mm=1.0,
        convex_scale_y_mm=1.0,
        convex_n_rays=5,
        convex_n_samples=4,
    )


def test_build_probe_geometry_from_args_supports_linear_and_convex():
    linear = build_probe_geometry_from_args(SimpleNamespace(probe_width=80.0, probe_depth=140.0, probe_type="linear"))
    assert linear.probe_type == "linear"

    convex = build_probe_geometry_from_args(
        SimpleNamespace(
            probe_width=80.0,
            probe_depth=140.0,
            probe_type="convex",
            convex_center_x=0.0,
            convex_center_y=0.0,
            convex_angle_deg=70.0,
            convex_outer_radius_px=860.0,
            convex_inner_radius_px=217.0,
            convex_scale_x_mm=0.233,
            convex_scale_y_mm=0.233,
            convex_n_rays=250,
            convex_n_samples=600,
            convex_sampling_strategy="uniform_fan",
        )
    )
    assert convex.is_convex
    assert convex.convex_render_shape == (600, 250)


def test_convex_ray_grid_uses_inner_arc_origins():
    geometry = make_convex_geometry()
    origins, directions = convex_ray_grid_local_mm(geometry)

    assert origins.shape == (5, 3)
    assert directions.shape == (5, 3)
    assert np.allclose(np.linalg.norm(directions[:, :2], axis=1), np.ones(5), atol=1e-6)
    assert np.allclose(np.linalg.norm(origins[:, :2], axis=1), np.full(5, geometry.convex_inner_radius_mm), atol=1e-6)


def test_convex_sampling_points_cover_expected_grid_shape():
    geometry = make_convex_geometry()
    points = convex_sampling_points_px(geometry)
    assert points.shape == (5, 4, 2)
    # center ray stays on the fan axis
    assert np.allclose(points[2, :, 0], np.full(4, geometry.convex_center_x), atol=1e-5)


def test_remap_image_to_convex_grid_and_back_preserves_centerline_signal():
    geometry = make_convex_geometry()
    image = np.zeros((80, 100), dtype=np.float32)
    image[20:50, 50] = 1.0

    fan = remap_image_to_convex_grid(image, geometry)
    restored = remap_convex_grid_to_image(fan, geometry, image.shape)

    assert fan.shape == (4, 5)
    assert restored.shape == image.shape
    assert float(restored[:, 50].max()) > 0.5
    rows = np.array([0.0, 0.0, 79.0], dtype=np.float32)
    cols = np.array([0.0, 99.0, 0.0], dtype=np.float32)
    outside_mask = ~convex_valid_pixel_mask(rows, cols, geometry)
    assert outside_mask.tolist() == [True, True, True]
    assert np.allclose(restored[0, 0], 0.0)


def test_remap_convex_grid_to_image_zeroes_pixels_outside_fan_support():
    geometry = make_convex_geometry()
    fan = np.ones((4, 5), dtype=np.float32)
    restored = remap_convex_grid_to_image(fan, geometry, (80, 100))

    mask = convex_valid_pixel_mask(
        *np.meshgrid(np.arange(80, dtype=np.float32), np.arange(100, dtype=np.float32), indexing="ij"),
        geometry,
    )
    assert float(restored[~mask].max()) == 0.0
    assert float(restored[mask].min()) > 0.0


def test_convex_valid_pixel_mask_excludes_points_outside_fan():
    geometry = make_convex_geometry()
    rows = np.array([10.0, 50.0, 10.0], dtype=np.float32)
    cols = np.array([50.0, 50.0, 0.0], dtype=np.float32)

    mask = convex_valid_pixel_mask(rows, cols, geometry)

    assert mask.tolist() == [False, True, False]

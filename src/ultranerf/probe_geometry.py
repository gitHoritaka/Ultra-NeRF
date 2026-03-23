"""Probe geometry definitions and convex sampling helpers.

This module is shared by training, rendering, and visualization code.
It keeps probe-type-specific logic out of the legacy visualization-only
transform helpers and avoids the legacy global convex configuration pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import griddata


ProbeType = str


@dataclass(frozen=True)
class ProbeGeometry:
    """Physical probe geometry and optional convex sampling metadata.

    All physical distances are expressed in millimeters. For convex probes, the
    geometry may additionally include image-space fan parameters in pixels and
    pixel-to-millimeter scaling factors.
    """

    width_mm: float
    depth_mm: float
    probe_type: ProbeType = "linear"
    convex_center_x: float | None = None
    convex_center_y: float | None = None
    convex_angle_deg: float | None = None
    convex_outer_radius_px: float | None = None
    convex_inner_radius_px: float | None = None
    convex_scale_x_mm: float | None = None
    convex_scale_y_mm: float | None = None
    convex_n_rays: int | None = None
    convex_n_samples: int | None = None
    convex_sampling_strategy: str = "uniform_fan"

    def __post_init__(self) -> None:
        object.__setattr__(self, "probe_type", str(self.probe_type).lower())
        if self.width_mm <= 0 or self.depth_mm <= 0:
            raise ValueError("width_mm and depth_mm must be strictly positive")
        if self.probe_type not in ("linear", "convex"):
            raise ValueError("probe_type must be either 'linear' or 'convex'")
        if self.probe_type == "convex":
            required = {
                "convex_center_x": self.convex_center_x,
                "convex_center_y": self.convex_center_y,
                "convex_angle_deg": self.convex_angle_deg,
                "convex_outer_radius_px": self.convex_outer_radius_px,
                "convex_inner_radius_px": self.convex_inner_radius_px,
                "convex_scale_x_mm": self.convex_scale_x_mm,
                "convex_scale_y_mm": self.convex_scale_y_mm,
                "convex_n_rays": self.convex_n_rays,
                "convex_n_samples": self.convex_n_samples,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"convex geometry is missing required fields: {', '.join(missing)}")
            if self.convex_outer_radius_px <= 0 or self.convex_inner_radius_px < 0:
                raise ValueError("convex radii must be non-negative and outer radius must be positive")
            if self.convex_outer_radius_px <= self.convex_inner_radius_px:
                raise ValueError("convex_outer_radius_px must be greater than convex_inner_radius_px")
            if self.convex_scale_x_mm <= 0 or self.convex_scale_y_mm <= 0:
                raise ValueError("convex pixel scales must be strictly positive")
            if self.convex_n_rays <= 0 or self.convex_n_samples <= 0:
                raise ValueError("convex_n_rays and convex_n_samples must be strictly positive")
            if self.convex_angle_deg <= 0 or self.convex_angle_deg > 360.0:
                raise ValueError("convex_angle_deg must be in the interval (0, 360]")

    @property
    def is_convex(self) -> bool:
        return self.probe_type == "convex"

    @property
    def convex_inner_radius_mm(self) -> float:
        if not self.is_convex:
            raise AttributeError("convex_inner_radius_mm is only defined for convex probes")
        return float(self.convex_inner_radius_px) * float(self.convex_scale_y_mm)

    @property
    def convex_outer_radius_mm(self) -> float:
        if not self.is_convex:
            raise AttributeError("convex_outer_radius_mm is only defined for convex probes")
        return float(self.convex_outer_radius_px) * float(self.convex_scale_y_mm)

    @property
    def convex_render_shape(self) -> tuple[int, int]:
        if not self.is_convex:
            raise AttributeError("convex_render_shape is only defined for convex probes")
        return int(self.convex_n_samples), int(self.convex_n_rays)


def build_probe_geometry_from_args(args: Any) -> ProbeGeometry:
    """Build a typed probe geometry object from parsed CLI/config args."""
    probe_type = str(getattr(args, "probe_type", "linear")).lower()
    common = {
        "width_mm": float(getattr(args, "probe_width")),
        "depth_mm": float(getattr(args, "probe_depth")),
        "probe_type": probe_type,
    }
    if probe_type == "linear":
        return ProbeGeometry(**common)
    return ProbeGeometry(
        **common,
        convex_center_x=float(getattr(args, "convex_center_x")),
        convex_center_y=float(getattr(args, "convex_center_y")),
        convex_angle_deg=float(getattr(args, "convex_angle_deg")),
        convex_outer_radius_px=float(getattr(args, "convex_outer_radius_px")),
        convex_inner_radius_px=float(getattr(args, "convex_inner_radius_px")),
        convex_scale_x_mm=float(getattr(args, "convex_scale_x_mm")),
        convex_scale_y_mm=float(getattr(args, "convex_scale_y_mm")),
        convex_n_rays=int(getattr(args, "convex_n_rays")),
        convex_n_samples=int(getattr(args, "convex_n_samples")),
        convex_sampling_strategy=str(getattr(args, "convex_sampling_strategy", "uniform_fan")),
    )


def convex_ray_angles_rad(geometry: ProbeGeometry) -> np.ndarray:
    """Return fan-ray angles in radians around the probe-local +Y axis."""
    if not geometry.is_convex:
        raise ValueError("convex_ray_angles_rad requires a convex ProbeGeometry")
    fan_half = np.deg2rad(float(geometry.convex_angle_deg)) * 0.5
    return np.linspace(-fan_half, fan_half, int(geometry.convex_n_rays), dtype=np.float32)


def convex_ray_grid_local_mm(geometry: ProbeGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Return convex ray origins and unit directions in probe-local millimeters."""
    angles = convex_ray_angles_rad(geometry)
    dirs = np.stack([np.sin(angles), np.cos(angles), np.zeros_like(angles)], axis=-1).astype(np.float32)
    origins = dirs.copy()
    origins[:, 0] *= geometry.convex_inner_radius_mm
    origins[:, 1] *= geometry.convex_inner_radius_mm
    return origins.astype(np.float32), dirs.astype(np.float32)


def convex_sampling_points_px(geometry: ProbeGeometry) -> np.ndarray:
    """Return original-image sampling points for a convex fan grid.

    The result has shape ``[n_rays, n_samples, 2]`` with coordinates stored as
    ``(col, row)`` floating pixel positions in the original input image.
    """
    if not geometry.is_convex:
        raise ValueError("convex_sampling_points_px requires a convex ProbeGeometry")
    origins_mm, dirs = convex_ray_grid_local_mm(geometry)
    radial_distance_mm = np.linspace(
        0.0,
        geometry.convex_outer_radius_mm - geometry.convex_inner_radius_mm,
        int(geometry.convex_n_samples),
        dtype=np.float32,
    )
    pts_mm = origins_mm[:, None, :] + dirs[:, None, :] * radial_distance_mm[None, :, None]
    cols = pts_mm[..., 0] / float(geometry.convex_scale_x_mm) + float(geometry.convex_center_x)
    rows = pts_mm[..., 1] / float(geometry.convex_scale_y_mm) + float(geometry.convex_center_y)
    return np.stack([cols, rows], axis=-1).astype(np.float32)


def sample_image_at_points(image: np.ndarray, sample_points_px: np.ndarray) -> np.ndarray:
    """Nearest-sample an image at floating pixel coordinates."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("image must be 2D")
    points = np.asarray(sample_points_px, dtype=np.float32)
    cols = np.rint(points[..., 0]).astype(np.int32)
    rows = np.rint(points[..., 1]).astype(np.int32)
    valid = (
        (rows >= 0)
        & (rows < img.shape[0])
        & (cols >= 0)
        & (cols < img.shape[1])
    )
    sampled = np.zeros(points.shape[:-1], dtype=np.float32)
    sampled[valid] = img[rows[valid], cols[valid]]
    return sampled.astype(np.float32)


def convex_valid_pixel_mask(
    row: np.ndarray | float,
    col: np.ndarray | float,
    geometry: ProbeGeometry,
) -> np.ndarray:
    """Return whether raw image pixels lie inside the convex fan support."""
    if not geometry.is_convex:
        raise ValueError("convex_valid_pixel_mask requires a convex ProbeGeometry")
    row_arr = np.asarray(row, dtype=np.float32)
    col_arr = np.asarray(col, dtype=np.float32)
    dx = (col_arr - float(geometry.convex_center_x)) * float(geometry.convex_scale_x_mm)
    dy = (row_arr - float(geometry.convex_center_y)) * float(geometry.convex_scale_y_mm)
    radius = np.sqrt(dx * dx + dy * dy)
    angle_deg = np.degrees(np.arctan2(dx, dy))
    half_angle = float(geometry.convex_angle_deg) * 0.5
    return (
        (radius >= float(geometry.convex_inner_radius_mm))
        & (radius <= float(geometry.convex_outer_radius_mm))
        & (angle_deg >= -half_angle)
        & (angle_deg <= half_angle)
    )


def remap_image_to_convex_grid(image: np.ndarray, geometry: ProbeGeometry) -> np.ndarray:
    """Project a raw convex image into the renderer's fan-grid layout."""
    sample_points = convex_sampling_points_px(geometry)
    sampled = sample_image_at_points(image, sample_points)
    return sampled.T.astype(np.float32)


def remap_convex_grid_to_image(
    image_grid: np.ndarray,
    geometry: ProbeGeometry,
    output_shape: tuple[int, int],
    *,
    method: str = "nearest",
) -> np.ndarray:
    """Project a convex fan-grid image back into the original image layout."""
    grid = np.asarray(image_grid, dtype=np.float32)
    if grid.ndim != 2:
        raise ValueError("image_grid must be 2D")
    sample_points = convex_sampling_points_px(geometry).reshape(-1, 2)
    values = grid.T.reshape(-1)
    rows = np.arange(output_shape[0], dtype=np.float32)
    cols = np.arange(output_shape[1], dtype=np.float32)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    query = np.stack([cc.reshape(-1), rr.reshape(-1)], axis=-1)
    remapped = griddata(sample_points, values, query, method=method, fill_value=0.0)
    if method != "nearest" and np.isnan(remapped).any():
        nearest = griddata(sample_points, values, query[np.isnan(remapped)], method="nearest", fill_value=0.0)
        remapped[np.isnan(remapped)] = nearest
    return remapped.reshape(output_shape).astype(np.float32)

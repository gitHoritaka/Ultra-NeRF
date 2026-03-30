"""Lazy runtime wrapper for arbitrary-pose NeRF rendering."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np

from ultranerf.probe_geometry import ProbeGeometry, build_probe_geometry_from_args, remap_convex_grid_to_image
from ultranerf.visualization.transforms import ensure_pose_matrix


class SupportsParseArgs(Protocol):
    def parse_args(self, args: list[str] | None = None) -> Any:
        ...


@dataclass(frozen=True)
class NerfRuntime:
    """Runtime components imported lazily from the existing training code."""

    torch: Any
    config_parser: Any
    create_nerf: Any
    render_us: Any


def import_nerf_runtime() -> NerfRuntime:
    """Import the existing PyTorch runtime lazily.

    This keeps the visualization backend importable in non-GUI and non-rendering
    contexts. The actual training/runtime modules are only imported when a
    session is created.
    """
    import torch

    from ultranerf.nerf_utils import create_nerf, render_us
    from ultranerf.unerf_config import config_parser

    return NerfRuntime(
        torch=torch,
        config_parser=config_parser,
        create_nerf=create_nerf,
        render_us=render_us,
    )


def pose_mm_to_model_pose_m(
    pose_probe_to_world_mm: np.ndarray,
    probe_geometry: ProbeGeometry | None = None,
) -> np.ndarray:
    """Convert a visualization-space pose in millimeters into model-space meters.

    Convex and linear poses both use the same public convention here: the pose
    translation is the displayed imaging origin. For convex probes, the runtime
    now consumes that inner-arc midpoint convention directly, so only the
    millimeter-to-meter conversion is applied.
    """
    pose_mm = ensure_pose_matrix(pose_probe_to_world_mm).astype(np.float32).copy()
    pose_mm[:3, 3] *= 0.001
    return pose_mm


@dataclass
class NerfSession:
    """Wrapper for arbitrary-pose rendering against the existing PyTorch runtime."""

    runtime: NerfRuntime
    args: Any
    device: Any
    image_shape: tuple[int, int]
    display_image_shape: tuple[int, int] | None
    probe_geometry: ProbeGeometry | None
    probe_width_mm: float
    probe_depth_mm: float
    render_kwargs: dict[str, Any]
    sw_m: float
    sh_m: float
    near_m: float
    far_m: float

    def _enrich_probe_geometry_override(self, probe_geometry_override: ProbeGeometry | None) -> ProbeGeometry | None:
        if probe_geometry_override is None:
            return None
        if not probe_geometry_override.is_convex:
            return probe_geometry_override
        if probe_geometry_override.convex_n_rays is not None and probe_geometry_override.convex_n_samples is not None:
            return self._rescale_convex_override_resolution(probe_geometry_override)

        base_geometry = self.probe_geometry
        if base_geometry is not None and base_geometry.is_convex:
            return ProbeGeometry(
                width_mm=float(probe_geometry_override.width_mm),
                depth_mm=float(probe_geometry_override.depth_mm),
                probe_type="convex",
                convex_center_x=float(probe_geometry_override.convex_center_x if probe_geometry_override.convex_center_x is not None else base_geometry.convex_center_x),
                convex_center_y=float(probe_geometry_override.convex_center_y if probe_geometry_override.convex_center_y is not None else base_geometry.convex_center_y),
                convex_angle_deg=float(probe_geometry_override.convex_angle_deg if probe_geometry_override.convex_angle_deg is not None else base_geometry.convex_angle_deg),
                convex_outer_radius_px=float(probe_geometry_override.convex_outer_radius_px if probe_geometry_override.convex_outer_radius_px is not None else base_geometry.convex_outer_radius_px),
                convex_inner_radius_px=float(probe_geometry_override.convex_inner_radius_px if probe_geometry_override.convex_inner_radius_px is not None else base_geometry.convex_inner_radius_px),
                convex_scale_x_mm=float(probe_geometry_override.convex_scale_x_mm if probe_geometry_override.convex_scale_x_mm is not None else base_geometry.convex_scale_x_mm),
                convex_scale_y_mm=float(probe_geometry_override.convex_scale_y_mm if probe_geometry_override.convex_scale_y_mm is not None else base_geometry.convex_scale_y_mm),
                convex_n_rays=int(base_geometry.convex_n_rays),
                convex_n_samples=int(base_geometry.convex_n_samples),
                convex_sampling_strategy=str(probe_geometry_override.convex_sampling_strategy or base_geometry.convex_sampling_strategy),
            )
        return self._rescale_convex_override_resolution(probe_geometry_override)

    def _rescale_convex_override_resolution(self, geometry: ProbeGeometry) -> ProbeGeometry:
        base_geometry = self.probe_geometry
        if base_geometry is None or not base_geometry.is_convex or not geometry.is_convex:
            return geometry
        if (
            int(geometry.convex_n_rays) != int(base_geometry.convex_n_rays)
            or int(geometry.convex_n_samples) != int(base_geometry.convex_n_samples)
        ):
            return geometry

        base_depth_mm = float(base_geometry.convex_outer_radius_mm - base_geometry.convex_inner_radius_mm)
        override_depth_mm = float(geometry.convex_outer_radius_mm - geometry.convex_inner_radius_mm)
        if base_depth_mm <= 0.0 or override_depth_mm <= 0.0 or float(base_geometry.convex_angle_deg) <= 0.0:
            return geometry

        angle_ratio = float(geometry.convex_angle_deg) / float(base_geometry.convex_angle_deg)
        depth_ratio = override_depth_mm / base_depth_mm
        if np.isclose(angle_ratio, 1.0, atol=1e-3) and np.isclose(depth_ratio, 1.0, atol=1e-3):
            return geometry

        scaled_n_rays = max(8, int(round(float(base_geometry.convex_n_rays) * angle_ratio)))
        scaled_n_samples = max(8, int(round(float(base_geometry.convex_n_samples) * depth_ratio)))
        return ProbeGeometry(
            width_mm=float(geometry.width_mm),
            depth_mm=float(geometry.depth_mm),
            probe_type="convex",
            convex_center_x=float(geometry.convex_center_x),
            convex_center_y=float(geometry.convex_center_y),
            convex_angle_deg=float(geometry.convex_angle_deg),
            convex_outer_radius_px=float(geometry.convex_outer_radius_px),
            convex_inner_radius_px=float(geometry.convex_inner_radius_px),
            convex_scale_x_mm=float(geometry.convex_scale_x_mm),
            convex_scale_y_mm=float(geometry.convex_scale_y_mm),
            convex_n_rays=scaled_n_rays,
            convex_n_samples=scaled_n_samples,
            convex_sampling_strategy=str(geometry.convex_sampling_strategy),
        )

    def _resolve_render_geometry(self, probe_geometry_override: ProbeGeometry | None) -> ProbeGeometry | None:
        return self._enrich_probe_geometry_override(probe_geometry_override) or self.probe_geometry

    def _resolve_render_shape(self, geometry: ProbeGeometry | None) -> tuple[int, int]:
        if geometry is not None and geometry.is_convex:
            return geometry.convex_render_shape
        if self.display_image_shape is not None:
            return (int(self.display_image_shape[0]), int(self.display_image_shape[1]))
        return self.image_shape

    def _resolve_render_spacing_and_far(self, geometry: ProbeGeometry | None, image_shape: tuple[int, int]) -> tuple[float, float, float]:
        if geometry is not None and geometry.is_convex:
            sw_m = float(geometry.convex_scale_x_mm) * 0.001
            sh_m = float(geometry.convex_scale_y_mm) * 0.001
            far_m = (float(geometry.convex_outer_radius_mm) - float(geometry.convex_inner_radius_mm)) * 0.001
            return sw_m, sh_m, far_m

        width_mm = float(geometry.width_mm) if geometry is not None else float(self.probe_width_mm)
        depth_mm = float(geometry.depth_mm) if geometry is not None else float(self.probe_depth_mm)
        sw_m = width_mm * 0.001 / float(image_shape[1])
        sh_m = depth_mm * 0.001 / float(image_shape[0])
        far_m = depth_mm * 0.001
        return sw_m, sh_m, far_m

    @staticmethod
    def _resolve_runtime_probe_geometry(args: Any, fallback_geometry: ProbeGeometry | None) -> ProbeGeometry | None:
        """Resolve the probe geometry encoded in the runtime config, if available."""
        try:
            return build_probe_geometry_from_args(args)
        except Exception:
            return fallback_geometry

    @classmethod
    def from_checkpoint(
        cls,
        *,
        config_path: str,
        checkpoint_path: str,
        image_shape: tuple[int, int],
        display_image_shape: tuple[int, int] | None = None,
        probe_geometry: ProbeGeometry | None = None,
        probe_width_mm: float,
        probe_depth_mm: float,
        device: str | None = None,
        runtime: NerfRuntime | None = None,
    ) -> "NerfSession":
        runtime = runtime or import_nerf_runtime()
        parser: SupportsParseArgs = runtime.config_parser()
        args = parser.parse_args(["--config", config_path, "--ft_path", checkpoint_path])
        runtime_device = runtime.torch.device(device or ("cuda" if runtime.torch.cuda.is_available() else "cpu"))

        _, render_kwargs_test, _, _, _ = runtime.create_nerf(args, device=runtime_device, mode="test")

        effective_geometry = cls._resolve_runtime_probe_geometry(args, probe_geometry)
        height, width = image_shape
        if effective_geometry is not None and effective_geometry.is_convex:
            height, width = effective_geometry.convex_render_shape
        probe_width_m = float(probe_width_mm) * 0.001
        probe_depth_m = float(probe_depth_mm) * 0.001
        if effective_geometry is not None and effective_geometry.is_convex:
            sw_m = float(effective_geometry.convex_scale_x_mm) * 0.001
            sh_m = float(effective_geometry.convex_scale_y_mm) * 0.001
            probe_depth_m = (effective_geometry.convex_outer_radius_mm - effective_geometry.convex_inner_radius_mm) * 0.001
        else:
            sw_m = probe_width_m / float(width)
            sh_m = probe_depth_m / float(height)
        near_m = 0.0
        far_m = probe_depth_m
        render_kwargs = dict(render_kwargs_test)
        render_kwargs.update({"near": near_m, "far": far_m})

        return cls(
            runtime=runtime,
            args=args,
            device=runtime_device,
            image_shape=(height, width),
            display_image_shape=display_image_shape,
            probe_geometry=effective_geometry,
            probe_width_mm=float(probe_width_mm),
            probe_depth_mm=float(probe_depth_mm),
            render_kwargs=render_kwargs,
            sw_m=sw_m,
            sh_m=sh_m,
            near_m=near_m,
            far_m=far_m,
        )

    def render_pose(self, pose_probe_to_world_mm: np.ndarray, **render_overrides: Any) -> dict[str, Any]:
        """Render the NeRF output for an arbitrary probe pose in millimeters."""
        probe_geometry_override = render_overrides.pop("probe_geometry_override", None)
        render_geometry = self._resolve_render_geometry(probe_geometry_override)
        image_shape = self._resolve_render_shape(render_geometry)
        sw_m, sh_m, far_m = self._resolve_render_spacing_and_far(render_geometry, image_shape)
        pose_m = pose_mm_to_model_pose_m(pose_probe_to_world_mm, probe_geometry=render_geometry)
        pose_tensor = self.runtime.torch.from_numpy(pose_m[:3, :4]).to(self.device).unsqueeze(0)
        kwargs = dict(self.render_kwargs)
        kwargs.update(render_overrides)
        kwargs["probe_geometry"] = render_geometry
        kwargs["near"] = self.near_m
        kwargs["far"] = far_m
        rendered = self.runtime.render_us(
            image_shape[0],
            image_shape[1],
            sw_m,
            sh_m,
            c2w=pose_tensor,
            chunk=self.args.chunk,
            **kwargs,
        )
        if isinstance(rendered, dict):
            rendered.setdefault("_render_mode", getattr(self.args, "render_mode", "default"))
            if render_geometry is not None:
                rendered.setdefault("_probe_geometry_type", render_geometry.probe_type)
        if render_geometry is not None and render_geometry.is_convex and self.display_image_shape is not None:
            return self._remap_convex_render_payload(rendered, render_geometry)
        return rendered

    def _remap_convex_render_payload(self, rendered_output: dict[str, Any], geometry: ProbeGeometry) -> dict[str, Any]:
        remapped: dict[str, Any] = {}
        for key, value in rendered_output.items():
            if str(key).startswith("_"):
                remapped[key] = value
                continue
            array = value
            try:
                np_value = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=np.float32)
            except Exception:
                remapped[key] = value
                continue
            squeezed = np.squeeze(np_value)
            if squeezed.ndim != 2:
                remapped[key] = value
                continue
            remapped_image = remap_convex_grid_to_image(
                squeezed,
                geometry,
                self.display_image_shape,
                method="bilinear",
            )
            tensor = self.runtime.torch.from_numpy(remapped_image).to(self.device).unsqueeze(0).unsqueeze(0)
            remapped[key] = tensor
        return remapped

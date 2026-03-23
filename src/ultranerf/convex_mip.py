"""Convex mip-style rendering backend.

This module ports the minimum viable convex MIP path from the legacy repo into
the current runtime without bringing over the legacy global config, full-volume
mode, or debug-only branches. The backend is convex-only and returns the same
acoustic maps as the default renderer by passing the network output through the
current acoustic integration path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultranerf.rendering import render_method_3


def mip_input_channels(multires: int) -> int:
    """Return the input width for integrated positional encoding."""
    return int(multires) * 2 * 3


def integrated_positional_encoding(
    means: torch.Tensor,
    covariances: torch.Tensor,
    *,
    multires: int,
) -> torch.Tensor:
    """Encode Gaussian samples with integrated positional encoding.

    Args:
        means: ``[N_rays, N_samples, 3]``
        covariances: ``[N_rays, N_samples, 3, 3]``
        multires: Number of mip-NeRF frequency bands.
    """
    if means.ndim != 3 or means.shape[-1] != 3:
        raise ValueError(f"means must have shape [N_rays, N_samples, 3], got {tuple(means.shape)}")
    if covariances.ndim != 4 or covariances.shape[-2:] != (3, 3):
        raise ValueError(
            "covariances must have shape [N_rays, N_samples, 3, 3], "
            f"got {tuple(covariances.shape)}"
        )

    device = means.device
    freqs = 2.0 ** torch.linspace(0.0, float(multires - 1), steps=int(multires), device=device)
    means_expanded = means.unsqueeze(-1)
    cov_diag = torch.diagonal(covariances, dim1=-2, dim2=-1).unsqueeze(-1)
    freq_grid = freqs.view(1, 1, 1, -1)

    phase = means_expanded * freq_grid
    attenuation = torch.exp(-0.5 * (freq_grid**2) * cov_diag)
    sin_encoding = torch.sin(phase) * attenuation
    cos_encoding = torch.cos(phase) * attenuation
    encoded = torch.stack([sin_encoding, cos_encoding], dim=-1)
    return encoded.flatten(-3)


def _compute_conical_frustum_gaussians(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    z_vals: torch.Tensor,
    *,
    pixel_radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate each conical frustum interval with a Gaussian."""
    n_rays, n_samples = z_vals.shape
    device = rays_o.device

    t0 = torch.cat([torch.zeros((n_rays, 1), device=device, dtype=z_vals.dtype), z_vals[:, :-1]], dim=-1)
    t1 = z_vals

    mu = torch.maximum((t0 + t1) * 0.5, torch.tensor(1e-10, device=device, dtype=z_vals.dtype))
    hw = torch.maximum((t1 - t0) * 0.5, torch.tensor(1e-10, device=device, dtype=z_vals.dtype))

    mu_sq = mu * mu
    hw_sq = hw * hw
    denom = torch.maximum(3.0 * mu_sq + hw_sq, torch.tensor(1e-10, device=device, dtype=z_vals.dtype))

    t_mean = mu + (2.0 * mu * hw_sq) / denom
    t_var = hw_sq / 3.0 - (4.0 / 15.0) * ((hw_sq * hw_sq * (12.0 * mu_sq - hw_sq)) / (denom * denom))
    t_var = torch.clamp(t_var, min=0.0)
    r_var = (pixel_radius**2) * (mu_sq / 4.0 + (5.0 / 12.0) * hw_sq - (4.0 / 15.0) * (hw_sq * hw_sq) / denom)
    r_var = torch.clamp(r_var, min=0.0)

    means = rays_o[:, None, :] + rays_d[:, None, :] * t_mean[..., None]

    rays_d_norm = F.normalize(rays_d, dim=-1)
    use_x_ref = torch.abs(rays_d_norm[:, 0]) < 0.9
    ref_vec = torch.zeros_like(rays_d_norm)
    ref_vec[use_x_ref, 0] = 1.0
    ref_vec[~use_x_ref, 1] = 1.0
    v1 = F.normalize(torch.cross(rays_d_norm, ref_vec, dim=-1), dim=-1, eps=1e-8)
    v2 = F.normalize(torch.cross(rays_d_norm, v1, dim=-1), dim=-1, eps=1e-8)
    basis = torch.stack([rays_d_norm, v1, v2], dim=-1)

    diag_vars = torch.zeros((n_rays, n_samples, 3, 3), device=device, dtype=z_vals.dtype)
    diag_vars[:, :, 0, 0] = t_var
    diag_vars[:, :, 1, 1] = r_var
    diag_vars[:, :, 2, 2] = r_var

    basis_expanded = basis[:, None, :, :]
    covariances = torch.matmul(torch.matmul(basis_expanded, diag_vars), basis_expanded.transpose(-2, -1))
    covariances = torch.nan_to_num(covariances, nan=1e-10, posinf=1e-10, neginf=1e-10)
    return means, covariances


def _compute_elongated_gaussians(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    z_vals: torch.Tensor,
    *,
    pixel_radius: float,
    center_point: torch.Tensor,
    max_elongation: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Experimental sideways-elongated Gaussian footprint."""
    means, covariances = _compute_conical_frustum_gaussians(
        rays_o,
        rays_d,
        z_vals,
        pixel_radius=pixel_radius,
    )
    device = means.device
    center = center_point.to(device=device, dtype=means.dtype)
    rays_d_norm = F.normalize(rays_d, dim=-1, eps=1e-8)

    horizontal = means[..., :2] - center[:2]
    horizontal_distance = torch.linalg.norm(horizontal, dim=-1)
    max_distance = torch.max(horizontal_distance)
    if float(max_distance) > 1e-8:
        normalized_distance = torch.clamp(horizontal_distance / max_distance, 0.0, 1.0)
    else:
        normalized_distance = torch.zeros_like(horizontal_distance)
    elongation = 1.0 + normalized_distance * (float(max_elongation) - 1.0)

    up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=means.dtype).expand_as(rays_d_norm)
    horizontal_dirs = rays_d_norm.clone()
    horizontal_dirs[:, 2] = 0.0
    horizontal_norm = torch.linalg.norm(horizontal_dirs, dim=-1, keepdim=True)
    fallback = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=means.dtype).expand_as(horizontal_dirs)
    horizontal_dirs = torch.where(horizontal_norm > 1e-6, horizontal_dirs / horizontal_norm, fallback)
    sideways = F.normalize(torch.cross(up, horizontal_dirs, dim=-1), dim=-1, eps=1e-8)

    use_x_ref = torch.abs(rays_d_norm[:, 0]) < 0.9
    ref_vec = torch.zeros_like(rays_d_norm)
    ref_vec[use_x_ref, 0] = 1.0
    ref_vec[~use_x_ref, 1] = 1.0
    v1 = F.normalize(torch.cross(rays_d_norm, ref_vec, dim=-1), dim=-1, eps=1e-8)
    v2 = F.normalize(torch.cross(rays_d_norm, v1, dim=-1), dim=-1, eps=1e-8)
    nearly_vertical = torch.abs(horizontal_norm.squeeze(-1)) <= 1e-6
    basis = torch.stack([rays_d_norm, sideways, up], dim=-1)
    basis[nearly_vertical, :, 1] = v1[nearly_vertical]
    basis[nearly_vertical, :, 2] = v2[nearly_vertical]
    basis = F.normalize(basis, dim=1, eps=1e-8)

    diag = torch.diagonal(covariances, dim1=-2, dim2=-1).clone()
    diag[:, :, 1] = diag[:, :, 1] * elongation
    diag_vars = torch.diag_embed(diag)
    basis_expanded = basis[:, None, :, :]
    covariances = torch.matmul(torch.matmul(basis_expanded, diag_vars), basis_expanded.transpose(-2, -1))
    covariances = torch.nan_to_num(covariances, nan=1e-10, posinf=1e-10, neginf=1e-10)
    return means, covariances


def render_rays_us_convex_mip(
    ray_batch: torch.Tensor,
    network_fn,
    network_query_fn,
    N_samples: int,
    *,
    multires: int,
    pixel_radius: float,
    use_elongation: bool = False,
    max_elongation: float = 5.0,
    center_point: torch.Tensor | None = None,
    lindisp: bool = False,
    **_: object,
) -> dict[str, torch.Tensor]:
    """Render ultrasound rays using a convex MIP-style integrated encoding path."""
    n_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    bounds = ray_batch[:, 6:8].reshape(-1, 1, 2)
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(0.0, 1.0, int(N_samples), device=ray_batch.device, dtype=ray_batch.dtype)
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand(n_rays, int(N_samples))

    if use_elongation:
        if center_point is None:
            center_point = torch.mean(rays_o, dim=0)
        means, covariances = _compute_elongated_gaussians(
            rays_o,
            rays_d,
            z_vals,
            pixel_radius=float(pixel_radius),
            center_point=center_point,
            max_elongation=float(max_elongation),
        )
    else:
        means, covariances = _compute_conical_frustum_gaussians(
            rays_o,
            rays_d,
            z_vals,
            pixel_radius=float(pixel_radius),
        )

    encoded = integrated_positional_encoding(means, covariances, multires=int(multires))
    raw = network_query_fn(encoded, network_fn)
    return render_method_3(raw)

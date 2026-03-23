from pathlib import Path

import numpy as np
import torch

from ultranerf.convex_mip import integrated_positional_encoding, mip_input_channels, render_rays_us_convex_mip
from ultranerf.model import NeRF
from ultranerf.nerf_utils import create_nerf, render_us
from ultranerf.probe_geometry import build_probe_geometry_from_args
from ultranerf.unerf_config import config_parser
from ultranerf.visualization.render_panel import format_render_metadata


def _make_convex_config_args(tmp_path: Path, *, render_mode: str = "convex_mip") -> object:
    basedir = tmp_path / "logs"
    expdir = basedir / "test_exp"
    expdir.mkdir(parents=True, exist_ok=True)
    parser = config_parser()
    return parser.parse_args(
        [
            "--expname",
            "test_exp",
            "--basedir",
            str(basedir),
            "--probe_type",
            "convex",
            "--probe_width",
            "120",
            "--probe_depth",
            "160",
            "--convex_center_x",
            "50",
            "--convex_center_y",
            "10",
            "--convex_angle_deg",
            "35",
            "--convex_outer_radius_px",
            "100",
            "--convex_inner_radius_px",
            "20",
            "--convex_scale_x_mm",
            "0.4",
            "--convex_scale_y_mm",
            "0.4",
            "--convex_n_rays",
            "8",
            "--convex_n_samples",
            "10",
            "--multires",
            "4",
            "--netdepth",
            "2",
            "--netwidth",
            "16",
            "--render_mode",
            render_mode,
            "--mip_use_elongation",
            "--mip_max_elongation",
            "3.5",
            "--mip_pixel_radius",
            "0.00173",
        ]
    )


def test_convex_mip_config_parser_accepts_grouped_flags(tmp_path: Path) -> None:
    args = _make_convex_config_args(tmp_path)
    assert args.render_mode == "convex_mip"
    assert args.mip_use_elongation is True
    assert np.isclose(args.mip_max_elongation, 3.5)
    assert np.isclose(args.mip_pixel_radius, 0.00173)


def test_create_nerf_selects_convex_mip_backend_and_input_width(tmp_path: Path) -> None:
    args = _make_convex_config_args(tmp_path)
    render_kwargs_train, _, _, _, _ = create_nerf(args, device=torch.device("cpu"), mode="test")

    network = render_kwargs_train["network_fn"]
    assert isinstance(network, NeRF)
    assert network.input_ch == mip_input_channels(args.multires)
    assert render_kwargs_train["render_mode"] == "convex_mip"
    assert render_kwargs_train["mip_use_elongation"] is True


def test_create_nerf_rejects_convex_mip_for_linear_probe(tmp_path: Path) -> None:
    basedir = tmp_path / "logs"
    (basedir / "test_exp").mkdir(parents=True, exist_ok=True)
    parser = config_parser()
    args = parser.parse_args(
        [
            "--expname",
            "test_exp",
            "--basedir",
            str(basedir),
            "--probe_type",
            "linear",
            "--render_mode",
            "convex_mip",
        ]
    )
    try:
        create_nerf(args, device=torch.device("cpu"), mode="test")
    except ValueError as exc:
        assert "probe_type=convex" in str(exc)
    else:
        raise AssertionError("Expected convex_mip to reject linear probe_type")


def test_integrated_positional_encoding_shape_matches_contract() -> None:
    means = torch.zeros((2, 3, 3), dtype=torch.float32)
    covariances = torch.eye(3, dtype=torch.float32).reshape(1, 1, 3, 3).repeat(2, 3, 1, 1) * 0.1
    encoded = integrated_positional_encoding(means, covariances, multires=4)
    assert tuple(encoded.shape) == (2, 3, mip_input_channels(4))


def test_render_rays_us_convex_mip_returns_acoustic_maps() -> None:
    multires = 4
    model = NeRF(D=2, W=16, input_ch=mip_input_channels(multires), output_ch=3, skips=[]).cpu()

    def query_fn(inputs, network_fn):
        flat = inputs.reshape(-1, inputs.shape[-1])
        outputs = network_fn(flat)
        return outputs.reshape(*inputs.shape[:-1], outputs.shape[-1])

    rays_o = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.002, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    rays_d = torch.tensor([[0.0, 1.0, 0.0]] * 3, dtype=torch.float32)
    near = torch.zeros((3, 1), dtype=torch.float32)
    far = torch.ones((3, 1), dtype=torch.float32) * 0.03
    ray_batch = torch.cat([rays_o, rays_d, near, far], dim=-1)

    rendered = render_rays_us_convex_mip(
        ray_batch,
        network_fn=model,
        network_query_fn=query_fn,
        N_samples=5,
        multires=multires,
        pixel_radius=0.001,
        use_elongation=True,
        max_elongation=2.0,
    )

    assert "intensity_map" in rendered
    assert "confidence_maps" in rendered
    assert rendered["intensity_map"].ndim == 4
    assert set(rendered["intensity_map"].shape[-2:]) == {3, 5}


def test_render_metadata_includes_mip_mode_label() -> None:
    payload = {
        "_render_mode": "convex_mip",
        "intensity_map": torch.ones((1, 1, 4, 5), dtype=torch.float32),
    }
    metadata = format_render_metadata(payload)
    assert "Mode: convex_mip" in metadata


def test_convex_mip_runtime_smoke_render(tmp_path: Path) -> None:
    args = _make_convex_config_args(tmp_path)
    render_kwargs_train, _, _, _, _ = create_nerf(args, device=torch.device("cpu"), mode="test")
    geometry = build_probe_geometry_from_args(args)
    pose = torch.eye(4, dtype=torch.float32)[:3, :4].unsqueeze(0)

    rendered = render_us(
        geometry.convex_n_samples,
        geometry.convex_n_rays,
        float(geometry.convex_scale_x_mm) * 0.001,
        float(geometry.convex_scale_y_mm) * 0.001,
        c2w=pose,
        chunk=16,
        near=0.0,
        far=(geometry.convex_outer_radius_mm - geometry.convex_inner_radius_mm) * 0.001,
        **render_kwargs_train,
    )

    assert "intensity_map" in rendered
    assert "confidence_maps" in rendered
    assert rendered["intensity_map"].ndim == 4

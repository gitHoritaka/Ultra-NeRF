from pathlib import Path

import numpy as np

from ultranerf.probe_geometry import ProbeGeometry
from ultranerf.visualization.training_gui import GuiTrainingSessionController


def write_sweep(path: Path, value: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    images = np.full((2, 4, 5), value, dtype=np.float32)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    np.save(path / "images.npy", images)
    np.save(path / "poses.npy", poses)


class FakeProcess:
    def __init__(self):
        self._exit_code = None

    def poll(self):
        return self._exit_code


def test_controller_builds_preview_and_training_artifacts(tmp_path: Path) -> None:
    write_sweep(tmp_path / "sweep_a", 1.0)
    write_sweep(tmp_path / "sweep_b", 2.0)
    launched = {}

    def preview_launcher(path: Path):
        launched["preview_path"] = path

    controller = GuiTrainingSessionController(
        preview_launcher=preview_launcher,
        process_factory=lambda *args, **kwargs: FakeProcess(),
        run_root=tmp_path / "runs",
    )
    controller.discover_from_root(tmp_path)
    controller.set_selected_training_ids(("sweep_a",))
    controller.set_selected_validation_ids(("sweep_b",))
    controller.set_probe_geometry(ProbeGeometry(width_mm=20.0, depth_mm=30.0))
    controller.set_selected_scheme_path(Path("/workspace/configs/training_schemes/l2_baseline.json"))
    preview_manifest = controller.launch_preview()
    assert launched["preview_path"] == preview_manifest
    controller.set_preview_confirmed(True)
    artifacts = controller.start_training()
    assert artifacts.config_path.exists()
    assert artifacts.split_path.exists()
    assert artifacts.dataset_dir.exists()
    config_text = artifacts.config_path.read_text()
    assert "N_samples = 4" in config_text
    progress = controller.poll_progress()
    assert progress["event"] == "launched"


def test_reapplying_same_geometry_does_not_clear_preview_confirmation() -> None:
    controller = GuiTrainingSessionController(run_root=Path("/tmp/gui_training_test"))
    geometry = ProbeGeometry(width_mm=20.0, depth_mm=30.0)
    controller.set_probe_geometry(geometry)
    controller.set_preview_confirmed(True)
    controller.set_probe_geometry(ProbeGeometry(width_mm=20.0, depth_mm=30.0))
    assert controller.preview_confirmed is True


def test_convex_training_config_uses_convex_sample_count(tmp_path: Path) -> None:
    write_sweep(tmp_path / "sweep_a", 1.0)
    controller = GuiTrainingSessionController(
        process_factory=lambda *args, **kwargs: FakeProcess(),
        run_root=tmp_path / "runs",
    )
    controller.discover_from_root(tmp_path)
    controller.set_selected_training_ids(("sweep_a",))
    controller.set_selected_validation_ids(("sweep_a",))
    controller.set_probe_geometry(
        ProbeGeometry(
            width_mm=80.0,
            depth_mm=140.0,
            probe_type="convex",
            convex_center_x=100.0,
            convex_center_y=0.0,
            convex_angle_deg=70.0,
            convex_outer_radius_px=600.0,
            convex_inner_radius_px=100.0,
            convex_scale_x_mm=0.3,
            convex_scale_y_mm=0.3,
            convex_n_rays=250,
            convex_n_samples=384,
        )
    )
    controller.set_selected_scheme_path(Path("/workspace/configs/training_schemes/l2_baseline.json"))
    controller.launch_preview()
    controller.set_preview_confirmed(True)
    artifacts = controller.start_training()
    config_text = artifacts.config_path.read_text()
    assert "N_samples = 384" in config_text

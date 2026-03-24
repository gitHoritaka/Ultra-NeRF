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
    progress = controller.poll_progress()
    assert progress["event"] == "launched"

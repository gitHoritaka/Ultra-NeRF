"""GUI workflow for preparing and launching UltraNeRF training runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from ultranerf.probe_geometry import ProbeGeometry
from ultranerf.training_config import (
    DEFAULT_TRAINING_SCHEME_DIR,
    discover_training_scheme_files,
    load_training_scheme,
    write_flat_config,
)
from ultranerf.training_dataset import (
    DiscoveredTrainingSweep,
    build_preview_manifest,
    build_training_dataset_from_sweeps,
    discover_training_sweeps,
)


ProcessFactory = Callable[..., subprocess.Popen]
PreviewLauncher = Callable[[Path], Any]
ResultLauncher = Callable[[Path, Path, Path], Any]
VALIDATION_PREVIEW_MM_PER_PX = 0.4
VALIDATION_PREVIEW_SEPARATOR_PX = 24


def _default_process_factory(*args: Any, **kwargs: Any) -> subprocess.Popen:
    return subprocess.Popen(*args, **kwargs)


def _default_run_root() -> Path:
    return Path("logs") / "gui_training"


def _session_timestamp() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{int((time.time() % 1.0) * 1000):03d}"


def validation_preview_display_size_px(
    probe_geometry: ProbeGeometry,
    *,
    mm_per_px: float = VALIDATION_PREVIEW_MM_PER_PX,
    separator_px: int = VALIDATION_PREVIEW_SEPARATOR_PX,
) -> tuple[int, int]:
    """Return the target side-by-side preview display size in pixels.

    The preview shows target and render images side by side, so the total width
    is twice the physical image width plus a small separator.
    """
    if mm_per_px <= 0:
        raise ValueError("mm_per_px must be strictly positive")
    single_width = max(1, int(round(float(probe_geometry.width_mm) / mm_per_px)))
    height = max(1, int(round(float(probe_geometry.depth_mm) / mm_per_px)))
    return (single_width * 2) + int(separator_px), height


@dataclass
class TrainingRunArtifacts:
    """Paths and process handles for one launched GUI training run."""

    run_id: str
    run_dir: Path
    dataset_dir: Path
    split_path: Path
    config_path: Path
    stdout_log_path: Path
    preview_manifest_path: Path
    process: subprocess.Popen | None = None


class GuiTrainingSessionController:
    """Stateful backend for the GUI training workflow."""

    def __init__(
        self,
        *,
        preview_launcher: PreviewLauncher | None = None,
        result_launcher: ResultLauncher | None = None,
        process_factory: ProcessFactory | None = None,
        run_root: str | Path | None = None,
        scheme_root: str | Path | None = None,
    ) -> None:
        self.preview_launcher = preview_launcher
        self.result_launcher = result_launcher
        self.process_factory = process_factory or _default_process_factory
        self.run_root = _default_run_root() if run_root is None else Path(run_root)
        self.scheme_root = DEFAULT_TRAINING_SCHEME_DIR if scheme_root is None else Path(scheme_root)
        self.source_root: Path | None = None
        self.discovered_sweeps: tuple[DiscoveredTrainingSweep, ...] = ()
        self.selected_training_ids: tuple[str, ...] = ()
        self.selected_validation_ids: tuple[str, ...] = ()
        self.probe_geometry = ProbeGeometry(width_mm=80.0, depth_mm=140.0)
        self.preview_confirmed = False
        self.last_preview_manifest_path: Path | None = None
        self.selected_scheme_path: Path | None = None
        self.training_run: TrainingRunArtifacts | None = None
        self._progress_line_count = 0
        self._latest_progress_event: dict[str, Any] | None = None
        self._latest_preview_path: Path | None = None

    def available_scheme_paths(self) -> tuple[Path, ...]:
        return discover_training_scheme_files(self.scheme_root)

    def discover_from_root(self, root_dir: str | Path) -> tuple[DiscoveredTrainingSweep, ...]:
        self.source_root = Path(root_dir)
        self.discovered_sweeps = discover_training_sweeps(self.source_root)
        self.selected_training_ids = ()
        self.selected_validation_ids = ()
        self.preview_confirmed = False
        self.last_preview_manifest_path = None
        return self.discovered_sweeps

    def set_selected_training_ids(self, sweep_ids: tuple[str, ...]) -> None:
        self.selected_training_ids = tuple(dict.fromkeys(str(sweep_id) for sweep_id in sweep_ids))
        self.preview_confirmed = False

    def set_selected_validation_ids(self, sweep_ids: tuple[str, ...]) -> None:
        self.selected_validation_ids = tuple(dict.fromkeys(str(sweep_id) for sweep_id in sweep_ids))
        self.preview_confirmed = False

    def set_probe_geometry(self, probe_geometry: ProbeGeometry) -> None:
        if probe_geometry == self.probe_geometry:
            self.probe_geometry = probe_geometry
            return
        self.probe_geometry = probe_geometry
        self.preview_confirmed = False

    def set_selected_scheme_path(self, path: str | Path | None) -> None:
        self.selected_scheme_path = None if path is None else Path(path)

    def set_preview_confirmed(self, confirmed: bool) -> None:
        self.preview_confirmed = bool(confirmed)

    def can_start_training(self) -> bool:
        return (
            bool(self.selected_training_ids)
            and bool(self.selected_validation_ids)
            and self.preview_confirmed
            and self.selected_scheme_path is not None
            and not self.is_training_running()
        )

    def is_training_running(self) -> bool:
        return self.training_run is not None and self.training_run.process is not None and self.training_run.process.poll() is None

    def prepare_preview_manifest(self) -> Path:
        if not self.selected_training_ids and not self.selected_validation_ids:
            raise ValueError("Select at least one sweep before opening a preview")
        all_selected = tuple(dict.fromkeys(self.selected_training_ids + self.selected_validation_ids))
        preview_dir = self.run_root / "preview_manifests"
        manifest_path = preview_dir / f"preview_{_session_timestamp()}.json"
        manifest = build_preview_manifest(
            self.discovered_sweeps,
            selected_ids=all_selected,
            probe_geometry=self.probe_geometry,
            output_path=manifest_path,
        )
        self.last_preview_manifest_path = manifest
        return manifest

    def launch_preview(self) -> Path:
        manifest_path = self.prepare_preview_manifest()
        if self.preview_launcher is not None:
            self.preview_launcher(manifest_path)
        return manifest_path

    def _build_generated_training_config(self, artifacts: dict[str, Path], *, run_id: str) -> dict[str, Any]:
        scheme_path = self.selected_scheme_path
        if scheme_path is None:
            raise RuntimeError("No training scheme has been selected")
        image_shape = None
        if self.discovered_sweeps:
            selected_ids = set(self.selected_training_ids + self.selected_validation_ids)
            for sweep in self.discovered_sweeps:
                if sweep.sweep_id in selected_ids:
                    image_shape = sweep.image_shape
                    break
        values: dict[str, Any] = {
            "expname": run_id,
            "basedir": str(self.run_root.resolve()),
            "datadir": str(artifacts["dataset_dir"].resolve()),
            "dataset_type": "us",
            "split_file": str(artifacts["split_path"].resolve()),
            "training_scheme": str(scheme_path.resolve()),
            "probe_type": self.probe_geometry.probe_type,
            "probe_width": float(self.probe_geometry.width_mm),
            "probe_depth": float(self.probe_geometry.depth_mm),
            "tensorboard": False,
        }
        if image_shape is not None and not self.probe_geometry.is_convex:
            values["N_samples"] = int(image_shape[0])
        if self.probe_geometry.is_convex:
            values.update(
                {
                    "convex_center_x": float(self.probe_geometry.convex_center_x),
                    "convex_center_y": float(self.probe_geometry.convex_center_y),
                    "convex_angle_deg": float(self.probe_geometry.convex_angle_deg),
                    "convex_outer_radius_px": float(self.probe_geometry.convex_outer_radius_px),
                    "convex_inner_radius_px": float(self.probe_geometry.convex_inner_radius_px),
                    "convex_scale_x_mm": float(self.probe_geometry.convex_scale_x_mm),
                    "convex_scale_y_mm": float(self.probe_geometry.convex_scale_y_mm),
                    "convex_n_rays": int(self.probe_geometry.convex_n_rays),
                    "convex_n_samples": int(self.probe_geometry.convex_n_samples),
                    "N_samples": int(self.probe_geometry.convex_n_samples),
                }
            )
        return values

    def start_training(self) -> TrainingRunArtifacts:
        if not self.can_start_training():
            raise RuntimeError("Training cannot start until sweep roles, preview confirmation, and scheme selection are complete")
        run_id = f"gui_train_{_session_timestamp()}"
        run_dir = self.run_root / run_id
        dataset_dir = run_dir / "dataset"
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = build_training_dataset_from_sweeps(
            self.discovered_sweeps,
            training_ids=self.selected_training_ids,
            validation_ids=self.selected_validation_ids,
            output_dir=dataset_dir,
        )
        if self.last_preview_manifest_path is None:
            self.last_preview_manifest_path = self.prepare_preview_manifest()
        generated_config = self._build_generated_training_config(artifacts, run_id=run_id)
        config_path = write_flat_config(run_dir / "generated_training_config.txt", generated_config)
        stdout_log_path = run_dir / "stdout.log"
        stdout_handle = stdout_log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str((Path(__file__).resolve().parents[3] / "run_ultranerf.py").resolve()),
            "--config",
            str(config_path.resolve()),
        ]
        process = self.process_factory(command, cwd=str(Path(__file__).resolve().parents[3]), stdout=stdout_handle, stderr=subprocess.STDOUT)
        self.training_run = TrainingRunArtifacts(
            run_id=run_id,
            run_dir=run_dir,
            dataset_dir=Path(artifacts["dataset_dir"]),
            split_path=Path(artifacts["split_path"]),
            config_path=config_path,
            stdout_log_path=stdout_log_path,
            preview_manifest_path=self.last_preview_manifest_path,
            process=process,
        )
        self._progress_line_count = 0
        self._latest_progress_event = {"event": "launched", "run_id": run_id, "run_dir": str(run_dir.resolve())}
        self._latest_preview_path = None
        return self.training_run

    def poll_progress(self) -> dict[str, Any]:
        if self.training_run is None:
            return {"event": "idle"}
        progress_path = self.training_run.run_dir / "progress.jsonl"
        if progress_path.exists():
            lines = progress_path.read_text().splitlines()
            if len(lines) > self._progress_line_count:
                for line in lines[self._progress_line_count :]:
                    if line.strip():
                        self._latest_progress_event = json.loads(line)
                        preview_path = self._latest_progress_event.get("preview_path")
                        if preview_path:
                            self._latest_preview_path = Path(str(preview_path))
                self._progress_line_count = len(lines)
        exit_code = None if self.training_run.process is None else self.training_run.process.poll()
        payload = dict(self._latest_progress_event or {"event": "launched"})
        payload["run_dir"] = str(self.training_run.run_dir.resolve())
        payload["stdout_log_path"] = str(self.training_run.stdout_log_path.resolve())
        payload["is_running"] = exit_code is None
        payload["exit_code"] = exit_code
        return payload

    def load_latest_preview_image(self) -> np.ndarray | None:
        self.poll_progress()
        if self._latest_preview_path is None:
            return None
        image = Image.open(self._latest_preview_path).convert("L")
        return np.asarray(image, dtype=np.uint8)

    def latest_preview_path(self) -> Path | None:
        self.poll_progress()
        return self._latest_preview_path

    def latest_checkpoint_path(self) -> Path | None:
        if self.training_run is None:
            return None
        checkpoints = sorted(self.training_run.run_dir.glob("*.tar"))
        if not checkpoints:
            return None
        return checkpoints[-1]

    def can_open_result_visualization(self) -> bool:
        progress = self.poll_progress()
        if self.result_launcher is None:
            return False
        if bool(progress.get("is_running", False)):
            return False
        if int(progress.get("exit_code", -1)) != 0:
            return False
        if self.training_run is None:
            return False
        checkpoint_path = self.latest_checkpoint_path()
        return (
            checkpoint_path is not None
            and checkpoint_path.exists()
            and self.training_run.config_path.exists()
            and self.training_run.preview_manifest_path.exists()
        )

    def open_result_visualization(self) -> Any:
        if not self.can_open_result_visualization():
            raise RuntimeError("Completed training artifacts are not ready for visualization")
        if self.result_launcher is None:
            raise RuntimeError("No visualization launcher is configured")
        checkpoint_path = self.latest_checkpoint_path()
        if checkpoint_path is None:
            raise RuntimeError("No checkpoint was produced by the training run")
        return self.result_launcher(
            self.training_run.preview_manifest_path,
            checkpoint_path,
            self.training_run.config_path,
        )


class _FallbackTrainingLauncherWidget:
    """Headless placeholder used in tests without Qt."""

    def __init__(self, controller: GuiTrainingSessionController) -> None:
        self.controller = controller
        self.widget = object()


def create_training_launcher_widget(
    *,
    preview_launcher: PreviewLauncher | None = None,
    result_launcher: ResultLauncher | None = None,
    process_factory: ProcessFactory | None = None,
    run_root: str | Path | None = None,
) -> Any:
    """Create a Qt widget that opens the training workflow dialog."""
    try:
        from PyQt5.QtCore import QSize, Qt, QTimer
        from PyQt5.QtGui import QImage, QMovie, QPixmap
        from PyQt5.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDialog,
            QFileDialog,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QSpinBox,
            QDoubleSpinBox,
            QSizePolicy,
            QVBoxLayout,
            QWidget,
        )
    except Exception:
        return _FallbackTrainingLauncherWidget(
            GuiTrainingSessionController(
                preview_launcher=preview_launcher,
                result_launcher=result_launcher,
                process_factory=process_factory,
                run_root=run_root,
            )
        )

    if QApplication.instance() is None:
        return _FallbackTrainingLauncherWidget(
            GuiTrainingSessionController(
                preview_launcher=preview_launcher,
                result_launcher=result_launcher,
                process_factory=process_factory,
                run_root=run_root,
            )
        )

    # PyQt5 exposes QProgressBar, not ProgressBar.
    from PyQt5.QtWidgets import QProgressBar

    controller = GuiTrainingSessionController(
        preview_launcher=preview_launcher,
        result_launcher=result_launcher,
        process_factory=process_factory,
        run_root=run_root,
    )

    launcher_widget = QWidget()
    layout = QVBoxLayout(launcher_widget)
    layout.setContentsMargins(0, 0, 0, 0)
    title = QLabel("Training")
    open_button = QPushButton("Open Training")
    layout.addWidget(title)
    layout.addWidget(open_button)

    dialog = QDialog()
    dialog.setWindowTitle("UltraNeRF Training")
    dialog.setMinimumWidth(720)
    dialog_layout = QVBoxLayout(dialog)
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    dialog_layout.addWidget(scroll_area)
    content_widget = QWidget()
    content_widget.setMinimumWidth(680)
    scroll_area.setWidget(content_widget)
    content_layout = QVBoxLayout(content_widget)

    source_group = QGroupBox("Sweep Discovery")
    source_layout = QVBoxLayout(source_group)
    source_row = QHBoxLayout()
    root_edit = QLineEdit()
    browse_button = QPushButton("Browse")
    discover_button = QPushButton("Discover")
    source_row.addWidget(root_edit)
    source_row.addWidget(browse_button)
    source_row.addWidget(discover_button)
    source_layout.addLayout(source_row)
    discovered_label = QLabel("No sweeps discovered.")
    source_layout.addWidget(discovered_label)
    discovered_list = QListWidget()
    source_layout.addWidget(discovered_list)
    validation_combo = QComboBox()
    source_layout.addWidget(QLabel("Validation Sweep"))
    source_layout.addWidget(validation_combo)
    content_layout.addWidget(source_group)

    geometry_group = QGroupBox("Probe Geometry")
    geometry_form = QFormLayout(geometry_group)
    probe_type_combo = QComboBox()
    probe_type_combo.addItems(["linear", "convex"])
    width_spin = QDoubleSpinBox()
    width_spin.setRange(1.0, 500.0)
    width_spin.setValue(80.0)
    width_spin.setSuffix(" mm")
    depth_spin = QDoubleSpinBox()
    depth_spin.setRange(1.0, 500.0)
    depth_spin.setValue(140.0)
    depth_spin.setSuffix(" mm")
    angle_spin = QDoubleSpinBox()
    angle_spin.setRange(1.0, 360.0)
    angle_spin.setValue(70.0)
    angle_spin.setSuffix(" deg")
    inner_spin = QDoubleSpinBox()
    inner_spin.setRange(0.0, 5000.0)
    inner_spin.setValue(217.0)
    outer_spin = QDoubleSpinBox()
    outer_spin.setRange(1.0, 10000.0)
    outer_spin.setValue(860.0)
    scale_x_spin = QDoubleSpinBox()
    scale_x_spin.setRange(0.0001, 10.0)
    scale_x_spin.setDecimals(6)
    scale_x_spin.setValue(0.233)
    scale_x_spin.setSuffix(" mm/px")
    scale_y_spin = QDoubleSpinBox()
    scale_y_spin.setRange(0.0001, 10.0)
    scale_y_spin.setDecimals(6)
    scale_y_spin.setValue(0.233)
    scale_y_spin.setSuffix(" mm/px")
    center_x_spin = QDoubleSpinBox()
    center_x_spin.setRange(-10000.0, 10000.0)
    center_y_spin = QDoubleSpinBox()
    center_y_spin.setRange(-10000.0, 10000.0)
    center_y_spin.setValue(0.0)
    n_rays_spin = QSpinBox()
    n_rays_spin.setRange(16, 4096)
    n_rays_spin.setValue(250)
    n_samples_spin = QSpinBox()
    n_samples_spin.setRange(16, 4096)
    n_samples_spin.setValue(600)
    geometry_form.addRow("Probe Type", probe_type_combo)
    geometry_form.addRow("Width", width_spin)
    geometry_form.addRow("Depth", depth_spin)
    geometry_form.addRow("Opening Angle", angle_spin)
    geometry_form.addRow("Short Radius", inner_spin)
    geometry_form.addRow("Long Radius", outer_spin)
    geometry_form.addRow("Center X", center_x_spin)
    geometry_form.addRow("Center Y", center_y_spin)
    geometry_form.addRow("Scale X", scale_x_spin)
    geometry_form.addRow("Scale Y", scale_y_spin)
    geometry_form.addRow("Render Rays", n_rays_spin)
    geometry_form.addRow("Render Samples", n_samples_spin)
    preview_row = QHBoxLayout()
    preview_button = QPushButton("Open Preview")
    confirm_preview = QCheckBox("Geometry confirmed")
    preview_row.addWidget(preview_button)
    preview_row.addWidget(confirm_preview)
    geometry_form.addRow(preview_row)
    content_layout.addWidget(geometry_group)

    scheme_group = QGroupBox("Training Scheme")
    scheme_layout = QVBoxLayout(scheme_group)
    scheme_combo = QComboBox()
    scheme_description = QLabel("No scheme selected.")
    scheme_description.setWordWrap(True)
    scheme_layout.addWidget(scheme_combo)
    scheme_layout.addWidget(scheme_description)
    content_layout.addWidget(scheme_group)

    progress_group = QGroupBox("Training Progress")
    progress_layout = QVBoxLayout(progress_group)
    status_label = QLabel("Ready")
    progress_bar = QProgressBar()
    progress_bar.setRange(0, 100)
    preview_status_label = QLabel(
        f"No validation preview available. Display scale: {VALIDATION_PREVIEW_MM_PER_PX:.1f} mm/px"
    )
    preview_status_label.setWordWrap(True)
    preview_scroll = QScrollArea()
    preview_scroll.setWidgetResizable(True)
    preview_scroll.setMinimumHeight(320)
    preview_image_label = QLabel()
    preview_image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    preview_image_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    preview_scroll.setWidget(preview_image_label)
    train_button = QPushButton("Train")
    train_button.setEnabled(False)
    open_result_button = QPushButton("Open Result in Viewer")
    open_result_button.setEnabled(False)
    progress_layout.addWidget(status_label)
    progress_layout.addWidget(progress_bar)
    progress_layout.addWidget(preview_status_label)
    progress_layout.addWidget(preview_scroll)
    progress_layout.addWidget(train_button)
    progress_layout.addWidget(open_result_button)
    content_layout.addWidget(progress_group)
    content_layout.addStretch(1)

    timer = QTimer(dialog)
    timer.setInterval(1000)
    current_preview_movie: QMovie | None = None
    current_preview_path: Path | None = None

    def current_geometry() -> ProbeGeometry:
        probe_type = probe_type_combo.currentText()
        common = {
            "width_mm": float(width_spin.value()),
            "depth_mm": float(depth_spin.value()),
            "probe_type": probe_type,
        }
        if probe_type == "linear":
            return ProbeGeometry(**common)
        return ProbeGeometry(
            **common,
            convex_center_x=float(center_x_spin.value()),
            convex_center_y=float(center_y_spin.value()),
            convex_angle_deg=float(angle_spin.value()),
            convex_outer_radius_px=float(outer_spin.value()),
            convex_inner_radius_px=float(inner_spin.value()),
            convex_scale_x_mm=float(scale_x_spin.value()),
            convex_scale_y_mm=float(scale_y_spin.value()),
            convex_n_rays=int(n_rays_spin.value()),
            convex_n_samples=int(n_samples_spin.value()),
        )

    def refresh_train_button() -> None:
        train_button.setEnabled(controller.can_start_training())
        open_result_button.setEnabled(controller.can_open_result_visualization())

    def sync_selection_from_list() -> None:
        selected_training: list[str] = []
        for row in range(discovered_list.count()):
            item = discovered_list.item(row)
            if item.checkState():
                selected_training.append(str(item.data(256)))
        controller.set_selected_training_ids(tuple(selected_training))
        validation_id = validation_combo.currentData()
        controller.set_selected_validation_ids(()) if validation_id is None else controller.set_selected_validation_ids((str(validation_id),))
        refresh_train_button()

    def populate_discovered_sweeps() -> None:
        discovered_list.clear()
        validation_combo.clear()
        for sweep in controller.discovered_sweeps:
            item = QListWidgetItem(f"{sweep.sweep_id} | {sweep.frame_count} frames | {sweep.image_shape[0]}x{sweep.image_shape[1]}")
            item.setData(256, sweep.sweep_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            discovered_list.addItem(item)
            validation_combo.addItem(
                f"{sweep.sweep_id} | {sweep.frame_count} frames | {sweep.image_shape[0]}x{sweep.image_shape[1]}",
                sweep.sweep_id,
            )
        discovered_label.setText(f"Discovered {len(controller.discovered_sweeps)} compatible sweeps.")
        if validation_combo.count() > 0:
            validation_combo.setCurrentIndex(0)
        sync_selection_from_list()

    def refresh_schemes() -> None:
        scheme_combo.clear()
        for scheme_path in controller.available_scheme_paths():
            scheme_combo.addItem(scheme_path.stem, scheme_path)
        if scheme_combo.count() > 0:
            controller.set_selected_scheme_path(Path(str(scheme_combo.currentData())))
            scheme = load_training_scheme(Path(str(scheme_combo.currentData())))
            steps = scheme.training_overrides.get("n_iters")
            suffix = "" if steps is None else f" | steps: {steps}"
            scheme_description.setText(f"{scheme.description}{suffix}")

    def choose_root() -> None:
        selected = QFileDialog.getExistingDirectory(dialog, "Select sweep root", root_edit.text() or str(Path.cwd()))
        if selected:
            root_edit.setText(selected)

    def discover_clicked() -> None:
        try:
            controller.discover_from_root(root_edit.text())
        except Exception as exc:
            QMessageBox.critical(dialog, "Sweep discovery failed", str(exc))
            return
        populate_discovered_sweeps()

    def scheme_changed() -> None:
        data = scheme_combo.currentData()
        if data is None:
            controller.set_selected_scheme_path(None)
            scheme_description.setText("No scheme selected.")
        else:
            scheme_path = Path(str(data))
            controller.set_selected_scheme_path(scheme_path)
            scheme = load_training_scheme(scheme_path)
            steps = scheme.training_overrides.get("n_iters")
            suffix = "" if steps is None else f" | steps: {steps}"
            scheme_description.setText(f"{scheme.description}{suffix}")
        refresh_train_button()

    def geometry_changed() -> None:
        is_convex = probe_type_combo.currentText() == "convex"
        for widget in (
            angle_spin,
            inner_spin,
            outer_spin,
            center_x_spin,
            center_y_spin,
            scale_x_spin,
            scale_y_spin,
            n_rays_spin,
            n_samples_spin,
        ):
            widget.setEnabled(is_convex)
        controller.set_probe_geometry(current_geometry())
        confirm_preview.setChecked(False)
        refresh_train_button()

    def preview_clicked() -> None:
        controller.set_probe_geometry(current_geometry())
        try:
            controller.launch_preview()
            status_label.setText("Preview opened. Confirm geometry before training.")
        except Exception as exc:
            QMessageBox.critical(dialog, "Preview failed", str(exc))
            return
        confirm_preview.setChecked(False)
        refresh_train_button()

    def confirm_changed() -> None:
        controller.set_preview_confirmed(confirm_preview.isChecked())
        refresh_train_button()

    def update_progress_ui() -> None:
        nonlocal current_preview_movie, current_preview_path
        progress = controller.poll_progress()
        event = str(progress.get("event", "idle"))
        if event == "idle":
            return
        step = int(progress.get("step", 0))
        total_steps = max(int(progress.get("total_steps", 1)), 1)
        progress_bar.setValue(int((100.0 * step) / total_steps))
        status_label.setText(
            f"{event} | step {step}/{total_steps} | running={bool(progress.get('is_running', False))}"
        )
        latest_preview_path = controller.latest_preview_path()
        if latest_preview_path is not None and latest_preview_path.suffix.lower() == ".gif":
            preview_status_label.setText(
                str(latest_preview_path) if latest_preview_path is not None else "Validation preview available"
            )
            target_width, target_height = validation_preview_display_size_px(controller.probe_geometry)
            if current_preview_movie is None or current_preview_path != latest_preview_path:
                if current_preview_movie is not None:
                    current_preview_movie.stop()
                current_preview_movie = QMovie(str(latest_preview_path))
                current_preview_path = latest_preview_path
                preview_image_label.setMovie(current_preview_movie)
            current_preview_movie.setScaledSize(QSize(target_width, target_height))
            if current_preview_movie.state() != QMovie.Running:
                current_preview_movie.start()
            preview_image_label.resize(target_width, target_height)
        else:
            preview_image = controller.load_latest_preview_image()
            if current_preview_movie is not None:
                current_preview_movie.stop()
                current_preview_movie = None
                current_preview_path = None
                preview_image_label.setMovie(None)
            if preview_image is not None:
                preview_status_label.setText(
                    str(latest_preview_path) if latest_preview_path is not None else "Validation preview available"
                )
                height, width = preview_image.shape
                qimage = QImage(preview_image.data, width, height, width, QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(qimage)
                target_width, target_height = validation_preview_display_size_px(controller.probe_geometry)
                scaled = pixmap.scaled(
                    target_width,
                    target_height,
                    Qt.IgnoreAspectRatio,
                    Qt.SmoothTransformation,
                )
                preview_image_label.setPixmap(scaled)
                preview_image_label.resize(scaled.size())
            elif latest_preview_path is not None:
                preview_status_label.setText(f"Validation preview: {latest_preview_path}")
                preview_image_label.clear()
            else:
                preview_status_label.setText(
                    f"No validation preview available. Display scale: {VALIDATION_PREVIEW_MM_PER_PX:.1f} mm/px"
                )
                preview_image_label.clear()
        if progress.get("exit_code") is not None and not bool(progress.get("is_running", False)):
            timer.stop()
            if int(progress.get("exit_code")) == 0:
                status_label.setText(f"Training completed | run: {progress.get('run_dir')}")
            else:
                status_label.setText(f"Training failed (exit={progress.get('exit_code')}) | log: {progress.get('stdout_log_path')}")
            refresh_train_button()

    def train_clicked() -> None:
        try:
            artifacts = controller.start_training()
        except Exception as exc:
            QMessageBox.critical(dialog, "Training launch failed", str(exc))
            return
        status_label.setText(f"Training launched: {artifacts.run_dir}")
        progress_bar.setValue(0)
        timer.start()
        refresh_train_button()

    def open_result_clicked() -> None:
        try:
            controller.open_result_visualization()
        except Exception as exc:
            QMessageBox.critical(dialog, "Open result failed", str(exc))
            return
        timer.stop()
        dialog.close()

    def open_dialog() -> None:
        refresh_schemes()
        refresh_train_button()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    browse_button.clicked.connect(choose_root)
    discover_button.clicked.connect(discover_clicked)
    discovered_list.itemChanged.connect(lambda _item: sync_selection_from_list())
    validation_combo.currentIndexChanged.connect(lambda _index: sync_selection_from_list())
    scheme_combo.currentIndexChanged.connect(lambda _index: scheme_changed())
    for widget in (probe_type_combo, width_spin, depth_spin, angle_spin, inner_spin, outer_spin, center_x_spin, center_y_spin, scale_x_spin, scale_y_spin, n_rays_spin, n_samples_spin):
        widget.valueChanged.connect(lambda *_args: geometry_changed()) if hasattr(widget, "valueChanged") else widget.currentIndexChanged.connect(lambda *_args: geometry_changed())
    preview_button.clicked.connect(preview_clicked)
    confirm_preview.stateChanged.connect(lambda _state: confirm_changed())
    train_button.clicked.connect(train_clicked)
    open_result_button.clicked.connect(open_result_clicked)
    timer.timeout.connect(update_progress_ui)
    open_button.clicked.connect(open_dialog)
    refresh_schemes()
    geometry_changed()

    launcher_widget.controller = controller
    launcher_widget.dialog = dialog
    launcher_widget.open_button = open_button
    return launcher_widget

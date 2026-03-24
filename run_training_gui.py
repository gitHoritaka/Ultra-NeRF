"""Launch the GUI training workflow without requiring a pre-existing manifest."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if ROOT.name == "scripts":
    SRC = ROOT.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from ultranerf.visualization.app import NerfLaunchConfig
from ultranerf.visualization.multi_sweep_app import (
    launch_multi_sweep_visualization_app,
    prepare_multi_sweep_visualization_app,
    resolve_multi_sweep_render_image_shape,
)
from ultranerf.visualization.training_gui import create_training_launcher_widget


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the UltraNeRF GUI training workflow")
    parser.add_argument("--cache-root", type=str, default=None, help="Optional cache root for preview/result viewers")
    parser.add_argument(
        "--fusion-device",
        type=str,
        default="auto",
        help="Device for preview/result sweep fusion: auto, cpu, cuda, or cuda:<index>",
    )
    parser.add_argument(
        "--fusion-reduction",
        type=str,
        default="max",
        choices=("mean", "max"),
        help="How overlapping voxel contributions are combined during preview/result fusion",
    )
    parser.add_argument("--spacing-mm", type=float, nargs=3, default=(1.0, 1.0, 1.0), help="Preview/result voxel spacing in mm")
    parser.add_argument("--pixel-stride", type=int, nargs=2, default=(2, 2), help="Preview/result image sampling stride (row, col)")
    parser.add_argument(
        "--preset",
        type=str,
        default="soft_tissue",
        choices=("soft_tissue", "high_contrast", "sparse_signal"),
        help="Initial preview/result volume visualization preset",
    )
    parser.add_argument("--initial-pose-index", type=int, default=0, help="Initial recorded pose index in preview/result viewers")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        from PyQt5.QtWidgets import QApplication
    except ModuleNotFoundError as exc:
        raise SystemExit(f"PyQt5 is required for the training GUI: {exc}") from exc

    app = QApplication.instance() or QApplication(sys.argv)
    preview_sessions: list[object] = []

    def _close_preview_session(session: object) -> None:
        viewer_obj = getattr(session, "viewer", None)
        window = getattr(viewer_obj, "window", None)
        qt_window = getattr(window, "_qt_window", None)
        if qt_window is not None:
            try:
                qt_window.close()
            except Exception:
                pass

    def _prepare_state(manifest_path: Path):
        return prepare_multi_sweep_visualization_app(
            manifest_path=manifest_path,
            spacing_mm=tuple(float(v) for v in args.spacing_mm),
            pixel_stride=tuple(int(v) for v in args.pixel_stride),
            preset_name=args.preset,
            cache_root=args.cache_root,
            fusion_device=args.fusion_device,
            reduction_mode=args.fusion_reduction,
        )

    def _launch_training_preview(manifest_path: Path):
        while preview_sessions:
            _close_preview_session(preview_sessions.pop())
        preview_state = _prepare_state(manifest_path)
        session = launch_multi_sweep_visualization_app(
            preview_state,
            initial_pose_index=args.initial_pose_index,
            nerf_config=None,
        )
        preview_sessions.append(session)
        return session

    def _launch_training_result_visualization(
        manifest_path: Path,
        checkpoint_path: Path,
        config_path: Path,
    ):
        preview_state = _prepare_state(manifest_path)
        nerf_config = NerfLaunchConfig(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            trigger_mode="manual",
            render_image_shape=resolve_multi_sweep_render_image_shape(preview_state.scene),
        )
        session = launch_multi_sweep_visualization_app(
            preview_state,
            initial_pose_index=args.initial_pose_index,
            nerf_config=nerf_config,
        )
        preview_sessions.append(session)
        return session

    launcher_widget = create_training_launcher_widget(
        preview_launcher=_launch_training_preview,
        result_launcher=_launch_training_result_visualization,
    )

    dialog = getattr(launcher_widget, "dialog", None)
    if dialog is None:
        raise SystemExit("The training GUI could not be initialized in this environment")
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())

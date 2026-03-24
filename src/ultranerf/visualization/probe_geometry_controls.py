"""Probe geometry override controls for the visualization app."""

from __future__ import annotations

import math

from ultranerf.probe_geometry import ProbeGeometry


class ProbeGeometryControlsWidget:
    """Qt widget for overriding the render/display probe geometry."""

    def __init__(self, ui_controller) -> None:
        from PyQt5.QtWidgets import (
            QComboBox,
            QDoubleSpinBox,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QScrollArea,
            QVBoxLayout,
            QWidget,
        )

        self.ui_controller = ui_controller
        self._updating_fields = False

        outer_widget = QWidget()
        outer_layout = QVBoxLayout(outer_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(scroll_area)
        self.widget = outer_widget

        content_widget = QWidget()
        content_widget.setMinimumWidth(280)
        scroll_area.setWidget(content_widget)
        layout = QVBoxLayout(content_widget)
        layout.addWidget(QLabel("Probe Geometry"))
        self.status_label = QLabel("Using dataset geometry")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.probe_type_combo = QComboBox()
        self.probe_type_combo.addItem("Linear", "linear")
        self.probe_type_combo.addItem("Convex", "convex")
        form = QFormLayout()
        form.addRow("Probe Type", self.probe_type_combo)
        layout.addLayout(form)

        self.linear_group = QGroupBox("Linear Geometry")
        linear_form = QFormLayout(self.linear_group)
        self.linear_width_spin = self._make_double_spinbox(minimum=1.0, maximum=500.0)
        self.linear_depth_spin = self._make_double_spinbox(minimum=1.0, maximum=500.0)
        linear_form.addRow("Width (mm)", self.linear_width_spin)
        linear_form.addRow("Depth (mm)", self.linear_depth_spin)
        layout.addWidget(self.linear_group)

        self.convex_group = QGroupBox("Convex Geometry")
        convex_form = QFormLayout(self.convex_group)
        self.convex_angle_spin = self._make_double_spinbox(minimum=1.0, maximum=180.0)
        self.convex_depth_spin = self._make_double_spinbox(minimum=1.0, maximum=500.0)
        self.convex_short_radius_spin = self._make_double_spinbox(minimum=0.0, maximum=500.0)
        self.convex_long_radius_spin = self._make_double_spinbox(minimum=1.0, maximum=1000.0)
        convex_form.addRow("Opening Angle (deg)", self.convex_angle_spin)
        convex_form.addRow("Depth (mm)", self.convex_depth_spin)
        convex_form.addRow("Short Radius (mm)", self.convex_short_radius_spin)
        convex_form.addRow("Long Radius (mm)", self.convex_long_radius_spin)
        layout.addWidget(self.convex_group)

        button_row = QHBoxLayout()
        self.apply_button = QPushButton("Apply Geometry")
        self.reset_button = QPushButton("Use Dataset")
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.reset_button)
        layout.addLayout(button_row)
        layout.addStretch(1)

        self.probe_type_combo.currentIndexChanged.connect(self._update_visible_groups)
        self.apply_button.clicked.connect(self._apply_geometry)
        self.reset_button.clicked.connect(self._reset_geometry)
        self.convex_short_radius_spin.valueChanged.connect(self._handle_convex_radius_changed)
        self.convex_long_radius_spin.valueChanged.connect(self._handle_convex_radius_changed)
        self.convex_depth_spin.valueChanged.connect(self._handle_convex_depth_changed)
        self._update_visible_groups()

    @staticmethod
    def _make_double_spinbox(*, minimum: float, maximum: float) -> object:
        from PyQt5.QtWidgets import QDoubleSpinBox

        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(3)
        spin.setSingleStep(1.0)
        return spin

    def set_probe_geometry(self, geometry: ProbeGeometry, *, is_override: bool) -> None:
        self._updating_fields = True
        try:
            self.probe_type_combo.setCurrentIndex(0 if geometry.probe_type == "linear" else 1)
            if geometry.is_convex:
                self.convex_angle_spin.setValue(float(geometry.convex_angle_deg))
                self.convex_short_radius_spin.setValue(float(geometry.convex_inner_radius_mm))
                self.convex_long_radius_spin.setValue(float(geometry.convex_outer_radius_mm))
                self.convex_depth_spin.setValue(float(geometry.convex_outer_radius_mm - geometry.convex_inner_radius_mm))
            else:
                self.linear_width_spin.setValue(float(geometry.width_mm))
                self.linear_depth_spin.setValue(float(geometry.depth_mm))
            self.status_label.setText("Using override geometry" if is_override else "Using dataset geometry")
            self._update_visible_groups()
        finally:
            self._updating_fields = False

    def _update_visible_groups(self) -> None:
        probe_type = str(self.probe_type_combo.currentData())
        self.linear_group.setVisible(probe_type == "linear")
        self.convex_group.setVisible(probe_type == "convex")

    def _handle_convex_radius_changed(self, _value: float) -> None:
        if self._updating_fields:
            return
        self._updating_fields = True
        try:
            depth = max(0.0, self.convex_long_radius_spin.value() - self.convex_short_radius_spin.value())
            self.convex_depth_spin.setValue(depth)
        finally:
            self._updating_fields = False

    def _handle_convex_depth_changed(self, value: float) -> None:
        if self._updating_fields:
            return
        self._updating_fields = True
        try:
            self.convex_long_radius_spin.setValue(self.convex_short_radius_spin.value() + float(value))
        finally:
            self._updating_fields = False

    def _build_override_geometry(self) -> ProbeGeometry:
        current_geometry = self.ui_controller.get_effective_probe_geometry()
        probe_type = str(self.probe_type_combo.currentData())
        if probe_type == "linear":
            return ProbeGeometry(
                width_mm=float(self.linear_width_spin.value()),
                depth_mm=float(self.linear_depth_spin.value()),
                probe_type="linear",
            )

        base = current_geometry if current_geometry.is_convex else self.ui_controller.app_state.probe_geometry
        short_radius_mm = float(self.convex_short_radius_spin.value())
        long_radius_mm = float(self.convex_long_radius_spin.value())
        depth_mm = max(0.0, long_radius_mm - short_radius_mm)
        scale_y = float(base.convex_scale_y_mm)
        scale_x = float(base.convex_scale_x_mm)
        return ProbeGeometry(
            width_mm=float(2.0 * long_radius_mm * math.sin(math.radians(float(self.convex_angle_spin.value()) * 0.5))),
            depth_mm=float(depth_mm),
            probe_type="convex",
            convex_center_x=float(base.convex_center_x),
            convex_center_y=float(base.convex_center_y),
            convex_angle_deg=float(self.convex_angle_spin.value()),
            convex_outer_radius_px=float(long_radius_mm / scale_y),
            convex_inner_radius_px=float(short_radius_mm / scale_y),
            convex_scale_x_mm=scale_x,
            convex_scale_y_mm=scale_y,
            convex_n_rays=int(base.convex_n_rays),
            convex_n_samples=int(base.convex_n_samples),
            convex_sampling_strategy=str(base.convex_sampling_strategy),
        )

    def _apply_geometry(self) -> None:
        self.ui_controller.set_probe_geometry_override(self._build_override_geometry())

    def _reset_geometry(self) -> None:
        self.ui_controller.set_probe_geometry_override(None)


def create_probe_geometry_controls(ui_controller) -> ProbeGeometryControlsWidget:
    """Create the Qt probe-geometry-controls widget."""
    try:
        from PyQt5.QtWidgets import QApplication
    except ModuleNotFoundError:
        QApplication = None  # type: ignore[assignment]

    if QApplication is None or QApplication.instance() is None:
        class _FallbackProbeGeometryControls:
            def __init__(self, controller) -> None:
                self.ui_controller = controller
                self.widget = object()
                self.geometry = None
                self.is_override = False

            def set_probe_geometry(self, geometry: ProbeGeometry, *, is_override: bool) -> None:
                self.geometry = geometry
                self.is_override = bool(is_override)

        return _FallbackProbeGeometryControls(ui_controller)  # type: ignore[return-value]

    return ProbeGeometryControlsWidget(ui_controller)

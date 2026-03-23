# Visualization Tickets

This folder breaks the 3D sweep visualization and interactive probe rendering
work into implementation-sized tickets.

The intended end state is:

- tracked 2D ultrasound sweeps fused into a 3D volume
- transfer-function-based 3D visualization of the input data
- synchronized MPR views
- interactive probe placement and orientation control
- live NeRF rendering from arbitrary probe poses

## Recommended Delivery Order

1. [T01 - Define Coordinate Conventions](T01-define-coordinate-conventions.md)
2. [T02 - Build Transform Utilities](T02-build-transform-utilities.md)
3. [T03 - Fuse Sweeps Into a 3D Volume](T03-fuse-sweeps-into-volume.md)
4. [T04 - Add Volume Metadata and Caching](T04-add-volume-metadata-and-caching.md)
5. [T05 - Add a Basic 3D Volume Viewer](T05-add-basic-3d-volume-viewer.md)
6. [T06 - Add Synchronized MPR Views](T06-add-synchronized-mpr-views.md)
7. [T07 - Add Probe Representation](T07-add-probe-representation.md)
8. [T08 - Add Probe Placement From MPR](T08-add-probe-placement-from-mpr.md)
9. [T09 - Add Probe Orientation Controls](T09-add-probe-orientation-controls.md)
10. [T10 - Wrap NeRF Inference for Arbitrary Poses](T10-wrap-nerf-inference-for-arbitrary-poses.md)
11. [T11 - Connect Viewer Pose to NeRF Rendering](T11-connect-viewer-pose-to-nerf-rendering.md)
12. [T12 - Add Comparison View](T12-add-comparison-view.md)
13. [T13 - Add Trajectory Overlay](T13-add-trajectory-overlay.md)
14. [T14 - Add Transfer-Function Presets](T14-add-transfer-function-presets.md)
15. [T15 - Add CLI and Launch Entry Point](T15-add-cli-and-launch-entry-point.md)
16. [T16 - Add Tests for Visualization Backend](T16-add-tests-for-visualization-backend.md)
17. [T17 - Add Documentation and User Guide](T17-add-documentation-and-user-guide.md)
18. [T18 - Add NeRF Checkpoint Loading to the Visualization CLI](T18-add-nerf-checkpoint-loading-to-visualization-cli.md)
19. [T19 - Add a Rendered Output Panel to the Napari App](T19-add-rendered-output-panel-to-napari-app.md)
20. [T20 - Add Interactive Probe Manipulation Controls](T20-add-interactive-probe-manipulation-controls.md)
21. [T21 - Add Live Render Triggering and Throttling](T21-add-live-render-triggering-and-throttling.md)
22. [T22 - Add a Recorded-Frame Comparison Panel](T22-add-recorded-frame-comparison-panel.md)
23. [T23 - Add Workflow Docs and Manual QA for Interactive Rendering](T23-add-workflow-docs-and-manual-qa-for-interactive-rendering.md)

## Dependency Overview

- T01 is required before all geometry-heavy tickets.
- T02 depends on T01.
- T03 depends on T01 and T02.
- T04 depends on T03.
- T05 depends on T03 and T04.
- T06 depends on T05.
- T07 depends on T01 and T05.
- T08 depends on T06 and T07.
- T09 depends on T07.
- T10 depends on T01 and the existing rendering pipeline.
- T11 depends on T08, T09, and T10.
- T12 depends on T11.
- T13 depends on T05 and T07.
- T14 depends on T05.
- T15 depends on T05 through T11, depending on what is exposed in the first release.
- T16 should be added alongside T02, T03, T04, and T10, but can be tracked as a dedicated hardening ticket.
- T17 should be updated continuously, but closes after the feature becomes usable.
- T18 depends on T10 and T15.
- T19 depends on T18.
- T20 depends on T06, T08, T09, and T15.
- T21 depends on T18, T19, and T20.
- T22 depends on T12, T19, and T21.
- T23 depends on T18 through T22.

## Suggested Milestones

### Milestone 1: Static Sweep Visualization

- T01
- T02
- T03
- T04
- T05
- T13
- T14

### Milestone 2: Interactive Probe Placement

- T06
- T07
- T08
- T09

### Milestone 3: Live NeRF Interaction

- T10
- T11
- T12
- T15

### Milestone 4: Hardening and Documentation

- T16
- T17

### Milestone 5: Interactive NeRF Viewer

- T18
- T19
- T20
- T21
- T22
- T23

### Milestone 6: Multi-Sweep Visualization

- T24
- T25
- T26
- T27
- T28
- T29
- T30
- T31

### Milestone 7: Multi-Sweep App Integration

- T32
- T33
- T34
- T35

### Milestone 8: Convex Probe Support

- T36
- T37
- T38
- T39
- T40
- T41
- T42
- T43

### Milestone 9: Convex MIP Support

- T44
- T45
- T46
- T47
- T48

## Multi-Sweep Follow-Up Tickets

The tickets below extend the single-sweep viewer to support multiple tracked
acquisition sweeps captured from different angles.

Important assumption:

- if the NeRF was trained on the multi-sweep dataset successfully, the sweep
  poses should already align in a common world frame in theory
- even with that assumption, the visualization toolkit should explicitly check
  cross-sweep alignment before presenting fused overlays or cross-sweep nearest
  frame comparisons

Recommended order:

24. [T24 - Define the Multi-Sweep Data Model and Scene Contract](T24-define-multi-sweep-data-model-and-scene-contract.md)
25. [T25 - Add Multi-Sweep Dataset Loading and Manifest Support](T25-add-multi-sweep-dataset-loading-and-manifest-support.md)
26. [T26 - Add Cross-Sweep Alignment Validation](T26-add-cross-sweep-alignment-validation.md)
27. [T27 - Add Per-Sweep World Transform Support](T27-add-per-sweep-world-transform-support.md)
28. [T28 - Add Multi-Sweep Volume Fusion and Per-Sweep Overlays](T28-add-multi-sweep-volume-fusion-and-per-sweep-overlays.md)
29. [T29 - Add Sweep-Aware Comparison and Probe Context](T29-add-sweep-aware-comparison-and-probe-context.md)
30. [T30 - Add Multi-Sweep UI Controls and Viewer State Management](T30-add-multi-sweep-ui-controls-and-viewer-state-management.md)
31. [T31 - Add Multi-Sweep Workflow Docs and Manual QA](T31-add-multi-sweep-workflow-docs-and-manual-qa.md)

Additional dependency notes:

- T24 is required before all other multi-sweep tickets.
- T25 depends on T24.
- T26 depends on T24 and T25.
- T27 depends on T24 and should be implemented before any feature that assumes
  optional external registration transforms.
- T28 depends on T25, T26, and T27.
- T29 depends on T25 and T27.
- T30 depends on T28 and T29.
- T31 depends on T24 through T30.

## Multi-Sweep App Integration Tickets

These tickets cover the remaining work needed to make the multi-sweep backend
available through the running napari application.

Recommended order:

32. [T32 - Add a Multi-Sweep Launch Entry Point and Manifest CLI](T32-add-multi-sweep-launch-entry-point-and-manifest-cli.md)
33. [T33 - Add Multi-Sweep Scene Composition to Napari](T33-add-multi-sweep-scene-composition-to-napari.md)
34. [T34 - Wire Multi-Sweep Controls and Comparison Into the Live Viewer](T34-wire-multi-sweep-controls-and-comparison-into-the-live-viewer.md)
35. [T35 - Add End-to-End Multi-Sweep QA and Example Launch Workflows](T35-add-end-to-end-multi-sweep-qa-and-example-launch-workflows.md)

Additional dependency notes:

- T32 depends on T24 through T31.
- T33 depends on T28, T30, and T32.
- T34 depends on T29, T30, T32, and T33.
- T35 depends on T32 through T34.

## Convex Probe Support Tickets

These tickets cover adding convex-probe support to the current codebase by
porting the required geometry and rendering assumptions from
`legacy/convex_probe/` into the current `src/ultranerf/` layout.

Important implementation rule:

- do not merge the legacy convex repository wholesale
- instead, extract the convex-specific geometry, sampling, and visualization
  logic and integrate it behind the current repo's abstractions

Recommended order:

36. [T36 - Define the Convex Probe Geometry Contract](T36-define-convex-probe-geometry-contract.md)
37. [T37 - Add Convex Probe Configuration and CLI Support](T37-add-convex-probe-configuration-and-cli-support.md)
38. [T38 - Add Convex Ray Generation for Training and Inference](T38-add-convex-ray-generation-for-training-and-inference.md)
39. [T39 - Integrate Convex Rendering Layout Into the Active Renderer](T39-integrate-convex-rendering-layout-into-the-active-renderer.md)
40. [T40 - Add Convex Pixel-to-World Mapping for Visualization and Fusion](T40-add-convex-pixel-to-world-mapping-for-visualization-and-fusion.md)
41. [T41 - Add Convex Probe Representation to the Visualizer](T41-add-convex-probe-representation-to-the-visualizer.md)
42. [T42 - Add Convex Dataset and Viewer Workflow Support](T42-add-convex-dataset-and-viewer-workflow-support.md)
43. [T43 - Add Convex Regression Tests and Documentation](T43-add-convex-regression-tests-and-documentation.md)

Additional dependency notes:

- T36 is required before all other convex tickets.
- T37 depends on T36.
- T38 depends on T36 and T37.
- T39 depends on T38.
- T40 depends on T36 and should be implemented before convex sweep fusion or
  convex visual interpretation is trusted.
- T41 depends on T36 and T40.
- T42 depends on T37 through T41.
- T43 depends on T36 through T42.

## Convex MIP Support Tickets

These tickets cover selectively migrating the legacy convex MIP rendering path
after baseline convex support is already working in the current repo.

Important implementation rule:

- do not start MIP migration until the non-MIP convex path is working and
  tested
- keep MIP support optional and explicitly selectable

Recommended order:

44. [T44 - Define the Convex MIP Scope and Runtime Contract](T44-define-convex-mip-scope-and-runtime-contract.md)
45. [T45 - Port the Convex MIP Sampling and Rendering Backend](T45-port-the-convex-mip-sampling-and-rendering-backend.md)
46. [T46 - Add Convex MIP Configuration and CLI Wiring](T46-add-convex-mip-configuration-and-cli-wiring.md)
47. [T47 - Add Convex MIP Viewer and Output Support](T47-add-convex-mip-viewer-and-output-support.md)
48. [T48 - Add Convex MIP Regression Tests and Documentation](T48-add-convex-mip-regression-tests-and-documentation.md)

Additional dependency notes:

- T44 depends on T39 and should be completed before any MIP code is ported.
- T45 depends on T44.
- T46 depends on T44 and T45.
- T47 depends on T45 and T46.
- T48 depends on T44 through T47.

# Convex Support QA Checklist

Use this checklist after implementing the convex-probe and optional convex MIP
tickets.

The purpose of this checklist is to confirm that:

- baseline convex support is actually usable
- the visualizer behaves correctly with convex data
- linear workflows were not regressed
- optional MIP support remains isolated and explicit

## Baseline Convex Geometry

- [ ] `probe_type=convex` can be parsed from the current CLI/config flow.
- [ ] Required convex parameters are validated with clear errors when missing
      or invalid.
- [ ] Convex ray generation produces a fan geometry rather than a linear sheet.
- [ ] Convex rays have plausible origin placement and angular spread.
- [ ] Convex pixel-to-probe mapping places samples on a fan sector, not a
      rectangle.
- [ ] Convex scan-support geometry is represented consistently across training
      and visualization.

## Baseline Convex Rendering

- [ ] The main training/inference path can run in convex mode without shape or
      runtime errors.
- [ ] The active renderer returns a valid convex output image.
- [ ] The chosen convex output layout is consistent with the documented design.
- [ ] Intermediate acoustic maps are still available in convex mode when they
      are expected to be supported.
- [ ] Rendering a recorded training pose produces output that is structurally
      similar to the recorded frame.
- [ ] At least one quantitative check such as `MSE` / `MAE` is within an
      acceptable range for the chosen validation data.

## Convex Visualization and Fusion

- [ ] Convex sweeps can be loaded through the current visualizer entry points.
- [ ] Convex sweep fusion uses fan-sector geometry rather than rectangular
      linear-probe mapping.
- [ ] The fused convex volume is visually plausible in 3D.
- [ ] The convex probe overlay in the viewer shows a fan/sector rather than a
      linear plane.
- [ ] The convex probe overlay aligns with the fused volume and current probe
      pose.
- [ ] `Nearest Recorded Frame` still returns plausible matches in convex mode.
- [ ] `Snap To Nearest` still moves the probe to a sensible recorded pose in
      convex mode.
- [ ] The render-panel dropdown still lets the user switch between the final
      image and intermediate acoustic maps for the latest render.

## Multi-Sweep Convex Behavior

- [ ] Multi-sweep convex datasets can be loaded through the current manifest
      workflow.
- [ ] Cross-sweep alignment checks still run and report meaningful warnings or
      confirmation.
- [ ] Aggregate and per-sweep views remain usable with convex data.
- [ ] Visible/enabled sweep selection still works in convex mode.
- [ ] The viewer remains responsive enough for practical use in convex
      multi-sweep workflows.

## Linear Regression Safety

- [ ] Existing linear configs still parse unchanged.
- [ ] Existing linear training still runs.
- [ ] Existing linear visualization still opens and behaves correctly.
- [ ] Existing linear render outputs still work.
- [ ] Existing linear tests continue to pass.

## Legacy Convex Parity

- [ ] For at least one known convex example, the current repo and the legacy
      convex repo produce compatible ray geometry.
- [ ] For at least one known convex example, the current repo and the legacy
      convex repo produce compatible image-layout behavior.
- [ ] Any intentionally deferred legacy convex features are documented clearly.

## Optional Convex MIP Support

These checks apply only if the MIP tickets are implemented.

- [ ] MIP support is explicitly selectable and off by default.
- [ ] Enabling MIP selects the intended backend cleanly.
- [ ] Convex MIP rendering runs without breaking the baseline convex path.
- [ ] Convex MIP output shape/layout matches the documented contract.
- [ ] Viewer/output adapters can display convex MIP output correctly.
- [ ] Any MIP-specific limitations around intermediate maps are documented and
      visible to the user if relevant.
- [ ] Baseline convex behavior is unchanged when MIP is disabled.

## Documentation and Developer Clarity

- [ ] README links to the convex-support docs concisely.
- [ ] Detailed convex-support docs exist under `docs/`.
- [ ] The docs explain supported parameters, output layout, and current
      limitations.
- [ ] The docs state clearly which legacy convex and MIP features were
      intentionally not ported.

## Sign-Off

The convex implementation should not be considered complete until:

- [ ] baseline convex rendering works end to end
- [ ] convex visualization works end to end
- [ ] linear workflows remain intact
- [ ] optional MIP support, if implemented, is tested and documented

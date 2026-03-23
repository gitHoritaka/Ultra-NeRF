# Convex MIP Support

This repo now includes a first-pass `render_mode=convex_mip` backend.

## Runtime Contract

- `convex_mip` is an alternate render backend, not the default renderer with a
  hidden sampling flag.
- It is only supported with `probe_type=convex`.
- It keeps the current acoustic integration/output contract by feeding the MIP
  network output into the same `render_method_3()` acoustic renderer used by
  the default path.
- Viewer output handling stays the same because the backend returns the same
  image/map keys as the default renderer.

## What Changes In MIP Mode

- The network input changes from point samples plus positional encoding to
  Gaussian samples plus integrated positional encoding.
- The network input width becomes `6 * multires`.
- `render_mode=convex_mip` is explicit in config/CLI and remains off by
  default.

## Supported MIP Settings

- `render_mode = convex_mip`
- `mip_use_elongation`
- `mip_max_elongation`
- `mip_pixel_radius`

Example config:

- [`configs/config_convex_mip_example.txt`](../configs/config_convex_mip_example.txt)

## Current Scope

Included:

- convex-only MIP backend
- integrated positional encoding
- optional sideways elongation
- checkpoint-backed viewer rendering
- existing intermediate acoustic maps through the current render panel

Out of scope in this port:

- legacy full-volume convex branches
- legacy cache/debug plumbing
- legacy concentric/radial sampling alternatives
- exact legacy training parity guarantees

## Known Limitations

- reconstruction mode is not supported with `render_mode=convex_mip`
- the MIP path is intended for new checkpoints; old non-MIP convex checkpoints
  are not compatible because the network input width changes
- the viewer labels the render mode, but nearest-frame comparison still shows
  the original recorded frame, not a MIP-specific reference

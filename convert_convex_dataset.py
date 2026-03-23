"""Convert a tracked convex ultrasound dataset into UltraNeRF dataset files."""

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
import json

from ultranerf.convex_dataset_conversion import convert_convex_multi_sweep_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a raw convex tracking dataset into UltraNeRF format")
    parser.add_argument("--source-root", type=str, required=True, help="Directory containing sweep_* folders and fan_info.xml")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Directory to write converted sweeps and manifest. Defaults to <source-root>_ultranerf",
    )
    parser.add_argument("--fan-info-xml", type=str, default=None, help="Optional override for the fan geometry XML path")
    parser.add_argument(
        "--tracking-origin",
        type=str,
        default="image_center",
        choices=("image_center", "fan_center"),
        help="Point represented by the raw tracking pose translation",
    )
    parser.add_argument("--sweep-glob", type=str, default="sweep_*", help="Glob used to discover source sweep folders")
    parser.add_argument("--image-glob", type=str, default="*.png", help="Glob used to discover frame images in each sweep")
    parser.add_argument("--convex-n-rays", type=int, default=None, help="Override convex render ray count for the manifest")
    parser.add_argument("--convex-n-samples", type=int, default=None, help="Override convex render sample count for the manifest")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing converted files in the output root")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root) if args.output_root is not None else source_root.with_name(source_root.name + "_ultranerf")
    summary = convert_convex_multi_sweep_dataset(
        source_root=source_root,
        output_root=output_root,
        fan_info_xml=args.fan_info_xml,
        tracking_origin=args.tracking_origin,
        sweep_glob=args.sweep_glob,
        image_glob=args.image_glob,
        convex_n_rays=args.convex_n_rays,
        convex_n_samples=args.convex_n_samples,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

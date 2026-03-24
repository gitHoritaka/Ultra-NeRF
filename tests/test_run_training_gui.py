import run_training_gui


def test_training_gui_parser_accepts_default_arguments() -> None:
    parser = run_training_gui.build_parser()
    args = parser.parse_args([])
    assert args.fusion_device == "auto"
    assert tuple(args.spacing_mm) == (1.0, 1.0, 1.0)
    assert tuple(args.pixel_stride) == (2, 2)
    assert args.preset == "soft_tissue"

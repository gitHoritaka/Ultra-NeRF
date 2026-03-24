import json
from pathlib import Path

from ultranerf.training_config import (
    DatasetSplit,
    apply_training_scheme_overrides,
    load_training_scheme,
    resolve_dataset_split,
    write_dataset_split,
)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_load_training_scheme_reads_loss_terms_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "scheme.json"
    path.write_text(
        json.dumps(
            {
                "name": "Test Scheme",
                "description": "desc",
                "training_overrides": {"n_iters": 1234},
                "loss_terms": [{"name": "l2", "weight": 1.0}],
            }
        )
    )
    scheme = load_training_scheme(path)
    assert scheme.name == "Test Scheme"
    assert scheme.training_overrides["n_iters"] == 1234
    assert scheme.loss_terms[0].name == "l2"


def test_apply_training_scheme_overrides_updates_namespace() -> None:
    args = Namespace(n_iters=10, reg=True)
    scheme = load_training_scheme(Path("/workspace/configs/training_schemes/l2_baseline.json"))
    apply_training_scheme_overrides(args, scheme)
    assert args.n_iters == 3000
    assert args.reg is False
    assert args._training_scheme_name == "L2 Baseline"


def test_write_and_resolve_dataset_split_round_trip(tmp_path: Path) -> None:
    split = DatasetSplit(train_indices=(0, 1, 2), validation_indices=(3,), metadata={"a": 1})
    path = write_dataset_split(tmp_path / "split.json", split)
    loaded = resolve_dataset_split(path, frame_count=4)
    assert loaded is not None
    assert loaded.train_indices == (0, 1, 2)
    assert loaded.validation_indices == (3,)

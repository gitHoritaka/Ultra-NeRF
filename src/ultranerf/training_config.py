"""Shared helpers for declarative training config and scheme resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


DEFAULT_TRAINING_SCHEME_DIR = _repo_root() / "configs" / "training_schemes"


@dataclass(frozen=True)
class LossTerm:
    """One configured loss contribution in a training scheme."""

    name: str
    weight: float
    start_step: int = 0
    end_step: int | None = None
    key: str | None = None

    def is_active(self, step: int) -> bool:
        if step < int(self.start_step):
            return False
        if self.end_step is not None and step >= int(self.end_step):
            return False
        return True

    @property
    def resolved_key(self) -> str:
        return str(self.key or self.name)


@dataclass(frozen=True)
class TrainingScheme:
    """Declarative description of a reusable training recipe."""

    name: str
    description: str = ""
    training_overrides: dict[str, Any] = field(default_factory=dict)
    loss_terms: tuple[LossTerm, ...] = ()
    regularization: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def active_loss_terms(self, step: int) -> tuple[LossTerm, ...]:
        return tuple(term for term in self.loss_terms if term.is_active(step))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "training_overrides": dict(self.training_overrides),
            "loss_terms": [asdict(term) for term in self.loss_terms],
            "regularization": dict(self.regularization),
            "path": str(self.path.resolve()) if self.path is not None else None,
        }


@dataclass(frozen=True)
class DatasetSplit:
    """Explicit train/validation frame indices for a prepared dataset."""

    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "metadata": dict(self.metadata),
        }


def discover_training_scheme_files(root: str | Path | None = None) -> tuple[Path, ...]:
    """Return available training-scheme files."""
    base = DEFAULT_TRAINING_SCHEME_DIR if root is None else Path(root)
    if not base.exists():
        return ()
    return tuple(sorted(path for path in base.iterdir() if path.is_file() and path.suffix.lower() == ".json"))


def load_training_scheme(path: str | Path) -> TrainingScheme:
    """Load one JSON training-scheme file."""
    scheme_path = Path(path)
    payload = json.loads(scheme_path.read_text())
    loss_terms = tuple(
        LossTerm(
            name=str(entry["name"]),
            weight=float(entry["weight"]),
            start_step=int(entry.get("start_step", 0)),
            end_step=int(entry["end_step"]) if entry.get("end_step") is not None else None,
            key=str(entry["key"]) if entry.get("key") is not None else None,
        )
        for entry in payload.get("loss_terms", [])
    )
    if not loss_terms:
        raise ValueError(f"Training scheme {scheme_path} must define at least one loss term")
    return TrainingScheme(
        name=str(payload.get("name", scheme_path.stem)),
        description=str(payload.get("description", "")),
        training_overrides=dict(payload.get("training_overrides", {})),
        loss_terms=loss_terms,
        regularization=dict(payload.get("regularization", {})),
        path=scheme_path,
    )


def resolve_training_scheme(path: str | Path | None) -> TrainingScheme | None:
    """Resolve an optional scheme path against the repo defaults."""
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        candidate = DEFAULT_TRAINING_SCHEME_DIR / str(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Training scheme not found: {path}")
    return load_training_scheme(candidate)


def apply_training_scheme_overrides(args: Any, scheme: TrainingScheme | None) -> Any:
    """Mutate an argparse namespace-like object with scheme defaults."""
    if scheme is None:
        return args
    for key, value in scheme.training_overrides.items():
        if hasattr(args, key):
            setattr(args, key, value)
    for key, value in scheme.regularization.items():
        if hasattr(args, key):
            setattr(args, key, value)
    setattr(args, "_training_scheme", scheme)
    setattr(args, "_training_scheme_name", scheme.name)
    setattr(args, "_training_scheme_path", str(scheme.path.resolve()) if scheme.path is not None else None)
    return args


def resolve_dataset_split(path: str | Path | None, frame_count: int) -> DatasetSplit | None:
    """Load and validate an optional explicit dataset split file."""
    if path is None:
        return None
    split_path = Path(path)
    payload = json.loads(split_path.read_text())
    train_indices = tuple(int(index) for index in payload.get("train_indices", []))
    validation_indices = tuple(int(index) for index in payload.get("validation_indices", []))
    if not train_indices:
        raise ValueError(f"Dataset split {split_path} must define at least one train index")
    if not validation_indices:
        raise ValueError(f"Dataset split {split_path} must define at least one validation index")
    for index in train_indices + validation_indices:
        if index < 0 or index >= int(frame_count):
            raise ValueError(f"Dataset split index {index} is out of bounds for {frame_count} frames")
    return DatasetSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
        metadata=dict(payload.get("metadata", {})),
    )


def write_dataset_split(path: str | Path, split: DatasetSplit) -> Path:
    """Write one dataset-split file to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(split.to_json_dict(), indent=2))
    return output_path


def namespace_to_serializable_dict(args: Any) -> dict[str, Any]:
    """Convert an argparse namespace-like object into JSON-safe values."""
    payload: dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
        elif isinstance(value, (list, tuple)):
            payload[key] = list(value)
        else:
            payload[key] = str(value)
    return payload


def write_flat_config(path: str | Path, values: dict[str, Any]) -> Path:
    """Write a flat configargparse-style config file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key in sorted(values):
        value = values[key]
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                lines.append(f"{key} = True")
            continue
        if isinstance(value, (list, tuple)):
            joined = " ".join(str(item) for item in value)
            lines.append(f"{key} = {joined}")
            continue
        lines.append(f"{key} = {value}")
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def write_resolved_training_metadata(
    run_dir: str | Path,
    *,
    args: Any,
    training_scheme: TrainingScheme | None,
    dataset_split: DatasetSplit | None,
) -> Path:
    """Persist the resolved runtime configuration for reproducibility."""
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "resolved_args": namespace_to_serializable_dict(args),
        "training_scheme": training_scheme.to_json_dict() if training_scheme is not None else None,
        "dataset_split": None if dataset_split is None else dataset_split.to_json_dict(),
    }
    path = output_dir / "resolved_training_config.json"
    path.write_text(json.dumps(payload, indent=2))
    return path

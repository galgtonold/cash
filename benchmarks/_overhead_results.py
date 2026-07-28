"""Result types and JSON persistence for the overhead benchmark."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StatementMetric:
    code: str
    execution_time: float
    total_time: float
    status: str  # 'COMPUTED' | 'RESTORED' | 'SKIPPED' | 'UNKNOWN'
    cost_model_size_bytes: int | None = None
    cost_model_restore_seconds: float | None = None
    cost_model_type_name: str | None = None
    cost_model_family: str | None = None
    # Why a statement did not cache. Without these a warm run that restores
    # nothing is indistinguishable from one that had nothing worth restoring,
    # which is exactly how the sweep got misread once.
    uncacheable_reasons: list[str] = field(default_factory=list)
    skipped_reason: str | None = None
    storage: list[str] = field(default_factory=list)


@dataclass
class CellTiming:
    index: int
    notebook_cell_index: int
    wall_seconds: float
    source_chars: int
    statement_metrics: list[StatementMetric] = field(default_factory=list)


@dataclass
class RunResult:
    notebook: str
    mode: str  # 'off' | 'cold' | 'warm'
    repeat: int
    python_version: str
    cash_version: str
    platform: str
    cells: list[CellTiming]
    total_wall_seconds: float
    cache_dir_bytes: int


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def write_results(path: Path, result: RunResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(result), indent=2), encoding="utf-8")


def read_results(path: Path) -> RunResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = [
        CellTiming(
            index=c["index"],
            notebook_cell_index=c["notebook_cell_index"],
            wall_seconds=c["wall_seconds"],
            source_chars=c["source_chars"],
            statement_metrics=[
                StatementMetric(
                    code=m["code"],
                    execution_time=m["execution_time"],
                    total_time=m["total_time"],
                    status=m["status"],
                    cost_model_size_bytes=m.get("cost_model_size_bytes"),
                    cost_model_restore_seconds=m.get("cost_model_restore_seconds"),
                    cost_model_type_name=m.get("cost_model_type_name"),
                    cost_model_family=m.get("cost_model_family"),
                )
                for m in c["statement_metrics"]
            ],
        )
        for c in data["cells"]
    ]
    return RunResult(
        notebook=data["notebook"],
        mode=data["mode"],
        repeat=data["repeat"],
        python_version=data["python_version"],
        cash_version=data["cash_version"],
        platform=data["platform"],
        cells=cells,
        total_wall_seconds=data["total_wall_seconds"],
        cache_dir_bytes=data["cache_dir_bytes"],
    )

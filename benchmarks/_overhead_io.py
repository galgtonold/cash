"""Notebook IO for the overhead benchmark.

Loads code cells from .ipynb files and generates the synthetic micro
notebook used to isolate per-statement overhead from compute noise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeCell:
    """A code cell from a notebook.

    Attributes:
        index: Zero-based index among code cells only.
        notebook_cell_index: Zero-based index among all cells (code + markdown).
        source: Cell source as a single string (joined with no extra separator).
    """
    index: int
    notebook_cell_index: int
    source: str


def load_code_cells(path: Path) -> list[CodeCell]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[CodeCell] = []
    code_idx = 0
    for nb_idx, cell in enumerate(data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        out.append(CodeCell(index=code_idx, notebook_cell_index=nb_idx, source=src))
        code_idx += 1
    return out


def write_synthetic_micro(path: Path, n_statements: int = 100) -> None:
    """Write a notebook with one code cell containing ``n_statements`` tiny
    assignments. No imports, no I/O — the cell stays in the FileAccessTracker
    + capture_output path so we measure cash overhead with minimal compute.
    """
    lines = [f"a_{i} = {i} + 1" for i in range(n_statements)]
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": "\n".join(lines) + "\n",
            }
        ],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(path).write_text(json.dumps(nb, indent=1), encoding="utf-8")

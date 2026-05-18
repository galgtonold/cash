import json
from pathlib import Path

import pytest

from benchmarks._overhead_io import (
    CodeCell,
    load_code_cells,
    write_synthetic_micro,
)


def _make_nb(tmp_path: Path, cells: list[tuple[str, list[str]]]) -> Path:
    """Build a minimal .ipynb on disk; cells is [(cell_type, source_lines), ...]."""
    nb = {
        "cells": [
            {
                "cell_type": ctype,
                "metadata": {},
                "source": src,
                **({"outputs": [], "execution_count": None} if ctype == "code" else {}),
            }
            for ctype, src in cells
        ],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


def test_load_code_cells_returns_only_code_cells(tmp_path):
    path = _make_nb(tmp_path, [
        ("markdown", ["# title\n"]),
        ("code", ["x = 1\n", "y = 2\n"]),
        ("markdown", ["nope"]),
        ("code", ["z = x + y\n"]),
    ])
    cells = load_code_cells(path)
    assert [c.index for c in cells] == [0, 1]  # zero-based among code cells
    assert cells[0].source == "x = 1\ny = 2\n"
    assert cells[1].source == "z = x + y\n"


def test_load_code_cells_preserves_original_cell_index(tmp_path):
    path = _make_nb(tmp_path, [
        ("markdown", ["a"]),
        ("code", ["x = 1"]),
        ("markdown", ["b"]),
        ("code", ["y = 2"]),
    ])
    cells = load_code_cells(path)
    assert cells[0].notebook_cell_index == 1
    assert cells[1].notebook_cell_index == 3


def test_write_synthetic_micro_produces_loadable_notebook(tmp_path):
    out = tmp_path / "micro.ipynb"
    write_synthetic_micro(out, n_statements=10)

    cells = load_code_cells(out)
    assert len(cells) == 1
    lines = [line for line in cells[0].source.splitlines() if line.strip()]
    assert len(lines) == 10
    # Each line is a simple assignment with no I/O
    for line in lines:
        assert "=" in line
        assert "open(" not in line
        assert "read_" not in line


def test_synthetic_micro_default_size(tmp_path):
    out = tmp_path / "micro.ipynb"
    write_synthetic_micro(out)
    cells = load_code_cells(out)
    lines = [line for line in cells[0].source.splitlines() if line.strip()]
    assert len(lines) == 100  # default per spec

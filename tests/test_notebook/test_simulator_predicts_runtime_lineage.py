"""The upstream simulation predicts exactly the lineage the runtime records.

Two places used to value an ingredient by a formula of their own:

* the simulator's file component, when it has no record of what the
  statement itself read, hashed ``absolute path:mtime:size`` where the
  runtime hashes paths relative to the notebook, so a statement reading a
  file got a different lineage in the simulation than when it ran;
* the module invalidator hashed a changed module's raw bytes, where every
  other lineage carries the module's source identity, so a comment-only edit
  moved the module's lineage.

Each test runs the statement for real, then asks the simulation for the same
statement and compares.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from traitlets.config.configurable import Configurable

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.ipython.magics import CashMagics


class _Shell(Configurable):
    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type("MockDisplayPub", (), {"publish": MagicMock()})()


@pytest.fixture
def notebook(tmp_path):
    """Magics running cells of a notebook saved in *tmp_path*."""
    backend = InMemoryBackend()
    shell = _Shell()
    magics = CashMagics(shell, Cash(backend=backend, register_magic=False))
    magics._auto_cache_enabled = True
    nb_path = tmp_path / "analysis.ipynb"
    cells: list[str] = []

    def run(code):
        """Run *code* as a cell: a new one, or the existing cell holding it again."""
        if code not in cells:
            cells.append(code)
        nb_path.write_text(
            json.dumps(
                {
                    "cells": [
                        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c}
                        for c in cells
                    ],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 4,
                }
            ),
            encoding="utf-8",
        )
        with (
            patch("cash.notebook.upstream.checker.get_notebook_cells", side_effect=lambda _p=None: list(cells)),
            patch("cash.notebook.upstream.checker.get_notebook_cells_with_ids", return_value=None),
        ):
            magics._execute_cell(code)

    def simulate(code):
        """The lineages the simulation gives *code*'s outputs, from scratch."""
        return magics._upstream_checker.simulator.simulate_cell(code).virtual_lineage

    # The notebook's own directory: the runtime keys files relative to it.
    with patch("cash.notebook.statement.file_deps.get_notebook_path", return_value=str(nb_path)):
        yield magics, shell, backend, run, simulate
    backend.clear()
    shell.user_ns.clear()


class TestFileComponent:
    def test_the_simulation_values_a_read_file_as_the_runtime_does(self, notebook, tmp_path):
        magics, _shell, backend, run, simulate = notebook
        data = tmp_path / "data" / "rows.txt"
        data.parent.mkdir()
        data.write_text("a\nb\n", encoding="utf-8")
        code = f"TEXT = open({str(data)!r}).read()"

        run(code)
        state = magics._statement_processor.tracking_state
        runtime = state.variable_lineage["TEXT"]
        assert state.executed_file_deps.get("TEXT"), "the read was not tracked"

        # No stored entry and no record of the statement's own reads (a new
        # kernel's position): the simulation values the files the output
        # depends on.
        backend.clear()
        state.statement_file_reads.clear()
        assert simulate(code)["TEXT"] == runtime

    def test_an_edited_file_moves_both_alike(self, notebook, tmp_path):
        """The control: the prediction still follows the file."""
        magics, _shell, backend, run, simulate = notebook
        data = tmp_path / "rows.txt"
        data.write_text("a\n", encoding="utf-8")
        code = f"TEXT = open({str(data)!r}).read()"
        run(code)
        state = magics._statement_processor.tracking_state
        before = state.variable_lineage["TEXT"]

        data.write_text("a\nb\nc\n", encoding="utf-8")
        backend.clear()
        state.statement_file_reads.clear()
        predicted = simulate(code)["TEXT"]
        assert predicted != before

        run(code)
        assert state.variable_lineage["TEXT"] == predicted


@pytest.fixture
def local_module(tmp_path):
    name = f"simlib_{abs(hash(str(tmp_path))) % 10**8}"
    path = tmp_path / f"{name}.py"
    path.write_text("def scale(x):\n    return x * 2\n", encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        yield name, path
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(name, None)


def _rewrite(path, text):
    """Rewrite *path* so the tracker sees it move (new size and mtime)."""
    st = os.stat(path)
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


class TestModuleEdits:
    def test_the_invalidator_values_a_module_by_its_source_identity(self, local_module):
        from cash.notebook.lineage_formula import read_module_source_hash
        from cash.notebook.module_invalidator import ModuleInvalidator

        name, path = local_module
        tracker = MagicMock(dep_file_to_parents={})
        before = ModuleInvalidator._compute_module_lineage_hash(name, str(path), tracker)
        assert before == read_module_source_hash(str(path))

        _rewrite(path, "# tidied\ndef scale(x):\n    # doubles\n    return x * 2\n")
        assert ModuleInvalidator._compute_module_lineage_hash(name, str(path), tracker) == before, (
            "a comment-only edit moved the module's lineage"
        )

        _rewrite(path, "def scale(x):\n    return x * 3\n")
        assert ModuleInvalidator._compute_module_lineage_hash(name, str(path), tracker) != before

    def test_an_unreadable_module_is_marked_not_randomised(self, tmp_path):
        from cash.notebook.module_invalidator import ModuleInvalidator

        tracker = MagicMock(dep_file_to_parents={})
        missing = str(tmp_path / "gone.py")
        first = ModuleInvalidator._compute_module_lineage_hash("gone", missing, tracker)
        assert first == ModuleInvalidator._compute_module_lineage_hash("gone", missing, tracker)

    def test_a_comment_only_edit_moves_no_lineage(self, notebook, local_module):
        magics, _shell, _backend, run, simulate = notebook
        name, path = local_module
        state = magics._statement_processor.tracking_state

        run(f"import {name}")
        run(f"Y = {name}.scale(21)")
        module_before = state.variable_lineage[name]
        y_before = state.variable_lineage["Y"]

        _rewrite(path, "# tidied up\ndef scale(x):\n    # doubles it\n    return x * 2\n")
        run("Z = 1")  # the next cell notices the file moved

        assert state.variable_lineage[name] == module_before
        run(f"Y = {name}.scale(21)")
        assert state.variable_lineage["Y"] == y_before
        assert simulate(f"import {name}")[name] == module_before

    def test_a_real_edit_is_predicted_as_the_runtime_records_it(self, notebook, local_module):
        """The control: a change to the code moves the lineage, and the
        simulation arrives at the runtime's new value."""
        magics, _shell, _backend, run, simulate = notebook
        name, path = local_module
        state = magics._statement_processor.tracking_state

        run(f"import {name}")
        before = state.variable_lineage[name]

        _rewrite(path, "def scale(x):\n    return x * 3\n")
        run("Z = 1")
        run(f"import {name}")

        assert state.variable_lineage[name] != before
        assert simulate(f"import {name}")[name] == state.variable_lineage[name]

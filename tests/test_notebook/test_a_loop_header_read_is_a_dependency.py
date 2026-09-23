"""The two halves of the loop-header dependency bug, at unit level.

The integration twin is
``tests/test_notebook_integration/test_a_loop_header_read_is_a_dependency.py``
and carries the full story. Here each half is pinned on its own, because
they are in different modules and either one alone leaves the bug in place:

1. ``ForLoopHandler`` must record what the loop's HEADER read, so the file
   reaches the variables the loop mutated and, through them, the control
   structure's recorded outcome.
2. ``VirtualLineage`` must act on a moved file whether or not the recorded
   input lineages still match the simulated ones -- they routinely do not,
   for every name bound in the same cell as ``%cash_on``.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from traitlets.config.configurable import Configurable

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.ipython.magics import CashMagics


class MockShell(Configurable):
    """Mock IPython shell for testing."""

    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns


@pytest.fixture
def magics_fixture():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend
    backend.clear()
    shell.user_ns.clear()


def _run_cell(magics, code):
    """Execute *code* as the only cell of a one-cell notebook."""
    notebook_path = os.path.join(tempfile.mkdtemp(), "test.ipynb")
    with open(notebook_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "cells": [
                    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 4,
            },
            fh,
        )
    with (
        patch("cash.notebook.upstream.checker.get_notebook_cells") as cells,
        patch("cash.notebook.upstream.checker.get_notebook_cells_with_ids") as ids,
    ):
        cells.side_effect = lambda _path=None: [code]
        ids.return_value = None
        magics._execute_cell(code)


class TestTheHeaderIsRead:
    """Half one: ``ForLoopHandler`` records what ``node.iter`` opened."""

    def test_the_file_a_loop_header_reads_reaches_what_the_loop_built(self, magics_fixture, tmp_path):
        """``ALIAS`` depends on the file, though no body statement touched it.

        This is the assertion ``%cash_provenance ALIAS`` was failing: it
        reported ``Code: ALIAS = {}`` -- an empty dict with no inputs -- for a
        table whose every entry came out of a file.
        """
        magics, shell, _backend = magics_fixture
        data = tmp_path / "alias.tsv"
        data.write_text("LOC1\tGENE_A\nLOC2\tGENE_B\n", encoding="utf-8")
        shell.user_ns["DATA"] = str(data)

        _run_cell(
            magics,
            "ALIAS = {}\nfor line in open(DATA).read().splitlines():\n    o, n = line.split('\\t')\n    ALIAS[o] = n",
        )

        assert shell.user_ns["ALIAS"] == {"LOC1": "GENE_A", "LOC2": "GENE_B"}
        recorded = magics.tracking_state.executed_file_deps.get("ALIAS", set())
        assert any(os.path.samefile(p, data) for p in recorded if os.path.exists(p)), (
            "the loop's header was the only read of %s, and it reached "
            "nothing: ALIAS's file deps are %r" % (data, sorted(recorded))
        )

    def test_a_header_that_reads_nothing_records_nothing(self, magics_fixture, tmp_path):
        """The control: an ordinary loop gains no dependency it has no business having."""
        magics, shell, _backend = magics_fixture
        shell.user_ns["RUNS"] = ["a", "b", "c"]

        _run_cell(magics, "OUT = []\nfor r in RUNS:\n    OUT.append(r.upper())")

        assert shell.user_ns["OUT"] == ["A", "B", "C"]
        deps = magics.tracking_state.executed_file_deps.get("OUT", set())
        assert not [p for p in deps if os.path.exists(p)], sorted(deps)


class TestAMovedFileIsCheckedEitherWay:
    """Half two: the file check does not wait on the input-lineage comparison.

    A recorded outcome holds ``(input lineages, produced lineages, files,
    file component)``. Whether the files have moved is decided by the files;
    the input lineages have nothing to say about it. Gating the file check on
    them made it unreachable for every loop reading a name bound in the
    ``%cash_on`` cell -- which is where notebooks put their paths.
    """

    @staticmethod
    def _simulate(tmp_path, entry_lineages, files, stored_component):
        """Simulate one recorded loop and return ``vars_with_stale_files``.

        ``DATA`` is given a virtual lineage and, through *entry_lineages*,
        usually no recorded one: that gap IS the condition under test, so the
        helper makes it explicit rather than leaving it to coincidence.
        """
        import hashlib

        from cash.notebook._protocols import TrackingState
        from cash.notebook.upstream import NotebookSimulator

        shell = MockShell()
        shell.user_ns["DATA"] = str(tmp_path / "rows.txt")
        state = TrackingState()
        code = "for line in open(DATA).read().splitlines():\n    OUT[line] = len(line)"
        state.control_outcomes[hashlib.sha256(code.encode("utf-8")).hexdigest()] = (
            entry_lineages,
            {"OUT": "produced-lineage"},
            frozenset(files),
            stored_component,
        )
        simulator = NotebookSimulator(shell, Cash(backend=InMemoryBackend(), register_magic=False), state)
        return simulator.simulate_cell(code, {"DATA": "a-simulated-lineage"}).vars_with_stale_files

    def test_a_moved_file_is_stale_although_the_input_lineages_disagree(self, tmp_path):
        """The reported cell 0: ``DATA`` has no runtime lineage, so they never agree."""
        data = tmp_path / "rows.txt"
        data.write_text("aa\nbbb\n", encoding="utf-8")

        stale = self._simulate(
            tmp_path,
            # What the runtime recorded: nothing about DATA, because it was
            # bound in the cell that turned cash on.
            entry_lineages={},
            files=[str(data)],
            stored_component="a-component-from-before-the-file-moved",
        )
        assert "OUT" in stale, (
            "the file behind OUT moved and the simulation said nothing, "
            "because the recorded entry lineages did not match"
        )

    def test_an_unmoved_file_is_not_stale(self, tmp_path):
        """The control: the check must not fire on a file that sat still."""
        from cash.notebook.statement.file_deps import compute_file_hash_component

        data = tmp_path / "rows.txt"
        data.write_text("aa\nbbb\n", encoding="utf-8")

        stale = self._simulate(
            tmp_path,
            entry_lineages={},
            files=[str(data)],
            stored_component=compute_file_hash_component({str(data)}),
        )
        assert "OUT" not in stale, sorted(stale)

    def test_a_matching_outcome_with_its_files_in_place_is_still_trusted(self, tmp_path):
        """The control for the restructure itself.

        The file check moved in front of the input-lineage comparison, so the
        path that adopts the runtime's recorded lineages now hangs off an
        ``elif``. It has to keep doing exactly what it did -- that adoption is
        what stops a loop re-running after a restart.
        """
        import hashlib

        from cash.notebook._protocols import TrackingState
        from cash.notebook.statement.file_deps import compute_file_hash_component
        from cash.notebook.upstream import NotebookSimulator

        data = tmp_path / "rows.txt"
        data.write_text("aa\nbbb\n", encoding="utf-8")

        shell = MockShell()
        shell.user_ns["DATA"] = str(data)
        state = TrackingState()
        code = "for line in open(DATA).read().splitlines():\n    OUT[line] = len(line)"
        state.control_outcomes[hashlib.sha256(code.encode("utf-8")).hexdigest()] = (
            {"DATA": "a-simulated-lineage"},  # the entry lineages DO match
            {"OUT": "produced-lineage"},
            frozenset([str(data)]),
            compute_file_hash_component({str(data)}),
        )
        simulator = NotebookSimulator(shell, Cash(backend=InMemoryBackend(), register_magic=False), state)
        sim = simulator.simulate_cell(code, {"DATA": "a-simulated-lineage"})
        virtual, stale = sim.virtual_lineage, sim.vars_with_stale_files
        assert virtual.get("OUT") == "produced-lineage", virtual
        assert "OUT" not in stale, sorted(stale)

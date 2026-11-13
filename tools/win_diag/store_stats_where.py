"""Windows diagnostic: where does a store take the stats of the files it inherits?

For the windows-latest 3.14 failure of
tests/test_notebook/internals/test_a_store_below_an_unstored_read_stats_its_files_once.py::
test_a_statement_too_cheap_to_store_below_takes_no_stat_of_the_files
(``assert 400 <= 400 // 4``: ``Y = X[:]``, too cheap to store, took a stat
of each of the 400 files at its store). It passes on Linux 3.10/3.13/3.14
and on windows 3.10, where the store takes none.

Run from the repository root of a dev checkout (needs the test deps:
pytest and IPython, as the test does):

    set PYTHONPATH=src
    python <this file>

It runs that one test file with ``os.stat``/``os.lstat`` counted inside
``StatementStore.save`` (as the test's ``store_stats`` does) and prints,
for each test, the 6 commonest call stacks of those stats, and whether the
unstored read ``X = ...`` kept its records for the store below
(``records_as_run``) or was found moved (``inputs_moved``).

Each stack is prefixed with ``[the statement being saved]``. Expected on
a good build (as on Linux), for
``test_a_statement_too_cheap_to_store_below_takes_no_stat_of_the_files``:
400 stats, all under ``[e = (time.sleep(...``, the first entry stored
below, and none under ``[Y = X[:]]``. On the failing build the stacks
under ``[Y = X[:]]`` name the code that stats the 400 files: paste the
whole output. Likely suspects: ``inputs_moved`` of ``X = ...`` saying True on
NTFS (a file with a second hard link keeps its size and times per folder
entry), so ``note_ran`` kept no records and the store below re-stats
them; or ``_records_through_inputs``.
"""

from __future__ import annotations

import collections
import os
import sys

import pytest

TEST = "tests/test_notebook/internals/test_a_store_below_an_unstored_read_stats_its_files_once.py"


class Probe:
    def __init__(self) -> None:
        self.stacks: collections.Counter[str] = collections.Counter()
        self.moved: list[tuple[str, bool]] = []

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        from cash.notebook.statement.store import StatementStore

        real_stat, real_lstat = os.stat, os.lstat
        real_save, real_moved, real_ran = StatementStore.save, StatementStore.inputs_moved, StatementStore.note_ran
        inside = [0]
        saving = [""]
        probe = self

        def where() -> str:
            f, parts = sys._getframe(2), []
            while f is not None and len(parts) < 10:
                parts.append(f"{os.path.basename(f.f_code.co_filename)}:{f.f_code.co_name}:{f.f_lineno}")
                f = f.f_back
            return " <- ".join(parts)

        def stat(*a, **k):
            if inside[0]:
                probe.stacks[f"[{saving[0]}] {where()}"] += 1
            return real_stat(*a, **k)

        def lstat(*a, **k):
            if inside[0]:
                probe.stacks[f"[{saving[0]}] {where()}"] += 1
            return real_lstat(*a, **k)

        def save(self, run, *a, **k):
            inside[0] += 1
            saving[0] = (run.code.splitlines() or [""])[0][:30]
            try:
                return real_save(self, run, *a, **k)
            finally:
                inside[0] -= 1

        def inputs_moved(self, run, *a, **k):
            moved = real_moved(self, run, *a, **k)
            probe.moved.append(((run.code.splitlines() or [""])[0][:50], moved))
            return moved

        def note_ran(self, run, execution):
            out = real_ran(self, run, execution)
            kept = run.cache_key in self.tracking_state.records_as_run
            probe.moved.append(("note_ran " + (run.code.splitlines() or [""])[0][:42], kept))
            return out

        os.stat, os.lstat = stat, lstat
        StatementStore.save, StatementStore.inputs_moved, StatementStore.note_ran = save, inputs_moved, note_ran
        try:
            yield
        finally:
            os.stat, os.lstat = real_stat, real_lstat
            StatementStore.save, StatementStore.inputs_moved, StatementStore.note_ran = real_save, real_moved, real_ran
            print(f"\n=== {item.name}: stats in save: {sum(self.stacks.values())}")
            for stack, n in self.stacks.most_common(6):
                print(f"  {n:5d}  {stack}")
            for code, answer in self.moved:
                print(f"  inputs_moved/records kept: {answer!s:5}  {code}")
            self.stacks.clear()
            self.moved.clear()


if __name__ == "__main__":
    print(sys.version, sys.platform)
    sys.exit(pytest.main(["-p", "no:cacheprovider", "-n0", "-s", "-q", TEST], plugins=[Probe()]))

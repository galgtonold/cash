"""A call statement's argument is read no more often on a rerun with new inputs, nor when it is polars.

Two ways the before-and-after hand-off (`DigestHandoff`) was missed, each
costing two more full reads of the argument per statement:

- A statement whose in-place-change verdict is known, run again on new
  inputs, fingerprints its arguments afresh (`MutationClassifier.start_rewatch`);
  those fingerprints were not handed to the call, so the call read the
  argument again each side. p21-polars-events re-ran ``sess = el.sessionize(events)``
  this way: four reads of a 674 MB frame, 9.6 s against 3.1 s plain.
- A polars frame had no memo, so each check of it read it in full. It is
  now read once per statement while its buffers stay where they were
  (`ArgFingerprints`): not across statements, since polars can share a
  numpy array's memory, and a write there would go unseen.
"""

from __future__ import annotations

import pytest

from cash import content_hashers
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")

SETUP = f"import time\ndef normalize(a):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return a * 2\n"


def _counting(monkeypatch, mock_shell, *names):
    seen = []
    for name in names:
        real = getattr(content_hashers, name)

        def counting(value, *a, _real=real, **k):
            if value is mock_shell.user_ns.get("x"):
                seen.append(1)
            return _real(value, *a, **k)

        monkeypatch.setattr(content_hashers, name, counting)
    return seen


MODULE = f"""import time


def normalize(a):
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    return a * 2
"""


def test_a_rerun_on_new_inputs_reads_the_argument_twice(cash_magics, mock_shell, monkeypatch, tmp_path):
    """The function lives in a module, so what it does to ``x`` is not read
    from its source: the verdict of the first run is checked again on the
    next run with new inputs."""
    (tmp_path / "rewatched_lib.py").write_text(MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    reads = _counting(monkeypatch, mock_shell, "hash_numpy")
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, "import numpy as np, rewatched_lib")
    run_cash_cell(cash_magics, "x = np.arange(300_000.0)")
    run_cash_cell(cash_magics, "y = rewatched_lib.normalize(x)")
    run_cash_cell(cash_magics, "x = np.arange(300_000.0) + 1")
    reads.clear()
    run_cash_cell(cash_magics, "y = rewatched_lib.normalize(x)")
    assert mock_shell.user_ns["y"][0] == 2.0
    # Was 4: the rewatch fingerprint before and after, and the call's hash
    # before and after it ran.
    assert len(reads) == 2, f"x was read {len(reads)} times"


def test_a_polars_argument_is_read_once(cash_magics, mock_shell, monkeypatch):
    pytest.importorskip("polars")
    reads = _counting(monkeypatch, mock_shell, "hash_polars")
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, "import polars as pl\n" + SETUP)
    run_cash_cell(cash_magics, "x = pl.DataFrame({'a': range(300_000), 'b': [1.5] * 300_000})")
    reads.clear()
    run_cash_cell(cash_magics, "y = normalize(x)")
    assert mock_shell.user_ns["y"]["a"][1] == 2
    # Was 3 or 4: the fingerprint before and after, the call's hash before
    # and after (the latter two shared already when nothing else runs).
    assert len(reads) == 1, f"x was read {len(reads)} times"


def test_a_polars_argument_changed_by_the_call_is_read_again(cash_magics, mock_shell, monkeypatch):
    pytest.importorskip("polars")
    reads = _counting(monkeypatch, mock_shell, "hash_polars")
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(
        cash_magics,
        "import time, polars as pl\n"
        "def stamp(a):\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    a[0, 'a'] = -1\n"
        "    return a.height\n",
    )
    run_cash_cell(cash_magics, "x = pl.DataFrame({'a': range(300_000), 'b': [1.5] * 300_000})")
    reads.clear()
    run_cash_cell(cash_magics, "n = stamp(x)")
    assert mock_shell.user_ns["x"]["a"][0] == -1
    assert len(reads) >= 2, "the write inside the call went unseen"

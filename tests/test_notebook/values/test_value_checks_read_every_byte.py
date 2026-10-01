"""Every notebook check that a value is unchanged reads the whole value.

A key built on a variable with no lineage, a call keyed on an argument it
computed, and the before/after check of a call's arguments all hashed a value
by its first rows, first elements or the ends of a list. Two values that
agreed there shared a key, and an edit past them read as no change: the
statement or call was served the other value's result.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from cash.notebook.call_interception import CallSite
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


def test_a_value_built_on_a_variable_without_lineage_moves_with_it(cash_magics, mock_shell):
    """A frame put into the namespace by something cash did not run (a
    ``%store -r``, a widget callback) has no lineage. A statement reading it
    runs every time, and what it produces is given a lineage from the frame's
    content -- so row 500 changing must give ``t`` a new lineage, or the
    cached ``u`` built on ``t`` is served for the old frame."""
    cash_magics.cash_on("")
    cash_magics.cash_persist("on")
    run_cash_cell(
        cash_magics,
        f"import time\ndef double(x):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return 2 * x",
    )
    frame = pd.DataFrame({"v": np.arange(1000, dtype=float)})
    mock_shell.user_ns["df"] = frame
    run_cash_cell(cash_magics, "t = float(df['v'].sum())")
    run_cash_cell(cash_magics, "u = double(t)")
    run_cash_cell(cash_magics, "u = double(t)")
    assert cash_magics.cash_status("dict")["last_cell"]["status"] != "COMPUTED", "control: unchanged is a hit"

    edited = frame.copy()
    edited.loc[500, "v"] = -1.0
    mock_shell.user_ns["df"] = edited
    run_cash_cell(cash_magics, "t = float(df['v'].sum())")
    run_cash_cell(cash_magics, "u = double(t)")
    assert mock_shell.user_ns["u"] == 2 * float(edited["v"].sum())


def _counted(tmp_path, body):
    """*body* wrapped to count its runs in a file, through ``os`` so the count
    is neither a global the callee writes (part of its key) nor a read."""
    log = str(tmp_path / "runs.log")

    def run(*args):
        fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        os.write(fd, b".")
        os.close(fd)
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return body(*args)

    def runs():
        return os.path.getsize(log) if os.path.exists(log) else 0

    return run, runs


def _normalise(rows):
    rows[500] = -1
    return len(rows)


def _score(batch):
    return float(batch.sum())


def test_a_call_that_edits_the_middle_of_a_long_list_is_not_cached(call_unit_harness, tmp_path):
    """``normalise(rows)`` rewrites one row in the middle of 1,000 and returns
    a count. Stored, a hit would return the count and leave ``rows`` as it was."""
    normalise, runs = _counted(tmp_path, _normalise)
    rows = list(range(1000))
    unit = call_unit_harness(lineage={"rows": "hash-rows"}, user_ns={"rows": rows, "normalise": normalise})
    site = CallSite(source="normalise(rows)", free_names=frozenset({"normalise", "rows"}), occurrence_index=0)

    unit.wrap(normalise, site)(rows)
    rows[500] = 500  # as a fresh run would find it
    unit.wrap(normalise, site)(rows)

    assert runs() == 2
    assert rows[500] == -1


def test_a_call_that_leaves_its_list_alone_is_cached(call_unit_harness, tmp_path):
    """Control for the test above: the same call without the edit is a hit."""
    count, runs = _counted(tmp_path, len)
    rows = list(range(1000))
    unit = call_unit_harness(lineage={"rows": "hash-rows"}, user_ns={"rows": rows, "count": count})
    site = CallSite(source="count(rows)", free_names=frozenset({"count", "rows"}), occurrence_index=0)

    unit.wrap(count, site)(rows)
    unit.wrap(count, site)(rows)

    assert runs() == 1


def test_calls_on_arrays_that_agree_in_their_first_elements_key_apart(call_unit_harness, tmp_path):
    """``score(next(batches))``: the argument is computed, so the call is keyed
    on its value. Two batches equal in their first 100 elements are still two
    batches; the same batch again is a hit."""
    score, runs = _counted(tmp_path, _score)
    # Object arrays: not plain data, so the call is keyed on its statement
    # and this digest, not on content alone.
    first = np.array([0] * 2000, dtype=object)
    second = first.copy()
    second[1000] = 5
    unit = call_unit_harness(lineage={"batches": "hash-batches"}, user_ns={"score": score})
    site = CallSite(
        source="score(next(batches))",
        free_names=frozenset({"score", "batches"}),
        occurrence_index=0,
        computed_arg_positions=(0,),
    )

    assert unit.wrap(score, site)(first) == 0.0
    assert unit.wrap(score, site)(first.copy()) == 0.0
    assert runs() == 1
    assert unit.wrap(score, site)(second) == 5.0


class _AnnDataLike:
    """The attributes scanpy writes to, as `mutation_fingerprint` sees an AnnData."""

    def __init__(self):
        self.obs = pd.DataFrame({"cluster": ["a", "b", "c", "a"]})
        self.var = pd.DataFrame({"gene": ["g1", "g2", "g3"]})
        self.X = np.arange(12, dtype=float).reshape(4, 3)
        self.uns = {"params": {"k": 10}}
        self.obsm = {"X_pca": np.zeros((4, 2))}
        self.varm = {}
        self.obsp = {}
        self.varp = {}
        self.layers = {}
        self.shape = (4, 3)


def test_an_anndata_like_value_edited_in_place_fingerprints_differently():
    """A bare ``sc.pp.something(adata)`` is learned as reading only when the
    fingerprint around it does not move. A column rewritten in place, rows of
    ``X`` swapped (same sum) or an embedding rescaled must all move it."""
    from cash.mutation_fingerprint import mutation_fingerprint

    adata = _AnnDataLike()
    before = mutation_fingerprint(adata)
    assert before is not None
    assert mutation_fingerprint(adata) == before

    adata.obs.loc[1, "cluster"] = "z"
    assert mutation_fingerprint(adata) != before

    adata = _AnnDataLike()
    adata.X[[0, 1]] = adata.X[[1, 0]]
    assert mutation_fingerprint(adata) != before

    adata = _AnnDataLike()
    adata.obsm["X_pca"][2, 1] = 3.0
    assert mutation_fingerprint(adata) != before

    adata = _AnnDataLike()
    adata.uns["params"]["k"] = 11
    assert mutation_fingerprint(adata) != before

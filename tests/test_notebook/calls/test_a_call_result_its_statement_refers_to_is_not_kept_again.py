"""A call result its statement refers to is digested without a second copy.

``faces = fetch_lfw_people(...)`` stores the call's result, and the
statement's entry refers to that entry (`call_refs`). The statement's digest
of ``faces`` still kept a copy of every buffer of its pickle, for a store of
the statement's that never takes them: 150 MB copied on every run, most of
what cash added to a 2 s restore.

What must hold: the digest is the one `compute_hash` gives the value, and
the call's own store still keeps its bytes.
"""

from __future__ import annotations

import numpy as np
import pytest

from cash import stored_pickle
from cash.value_hash import compute_hash
from tests._cell_driver import run_cash_cell


class Faces:
    """An object of a class of its own holding a few MB of arrays."""

    def __init__(self) -> None:
        self.images = np.ones((512, 1024))
        self.target = np.arange(4096)


SETUP = f"""
import time
from {__name__} import Faces

def fetch():
    time.sleep(0.3)
    return Faces()
"""


@pytest.fixture
def kept(cash_magics, monkeypatch):
    run_cash_cell(cash_magics, SETUP)
    keeps: list[bool] = []
    real = stored_pickle.pickle_value

    def counting(value, *args, **kwargs):
        if isinstance(value, Faces):
            keeps.append(bool(kwargs.get("keep")))
        return real(value, *args, **kwargs)

    monkeypatch.setattr(stored_pickle, "pickle_value", counting)
    return keeps


def test_only_the_call_s_store_keeps_its_bytes(cash_magics, mock_shell, kept):
    run_cash_cell(cash_magics, "faces = fetch()")
    assert kept.count(True) == 1, "the statement's digest kept a copy the call's store had made"
    faces = mock_shell.user_ns["faces"]
    assert cash_magics.tracking_state.current_session_hashes.get("faces") == compute_hash(faces)


def test_a_statement_that_holds_the_value_still_keeps_it(cash_magics, mock_shell, kept):
    run_cash_cell(cash_magics, "faces = [fetch()][0]")
    faces = mock_shell.user_ns["faces"]
    assert cash_magics.tracking_state.current_session_hashes.get("faces") == compute_hash(faces)

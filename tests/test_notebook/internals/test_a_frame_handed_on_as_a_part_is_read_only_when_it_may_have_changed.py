"""``t = f(x[x.b > 0])`` does not read ``x`` when the statement leaves it be.

The statement checks whether it changed ``x``. Its digest of ``x`` is
handed to no call (the call gets a part of it), so it was read in full
before the statement only to be compared with the check after it: 0.3 s
for a 5.5-million-row frame on p18's ``chg_tab``. Under copy-on-write the
frame is held instead (`ArgFingerprints.defer`) and read only when it no
longer provably is what it was (`ArgFingerprints.settle`). A frame handed
to the call as it is keeps its digest, which the call takes over
(`DigestHandoff`).
"""

from __future__ import annotations

import pytest

from cash import content_hashers
from cash.decorator import arg_hashing
from cash.notebook import call_effects
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


@pytest.fixture(autouse=True)
def _copy_on_write(monkeypatch, request):
    if int(pd.__version__.split(".", 1)[0]) < 3:
        before = pd.get_option("mode.copy_on_write")
        pd.set_option("mode.copy_on_write", True)
        request.addfinalizer(lambda: pd.set_option("mode.copy_on_write", before))
        monkeypatch.setattr(arg_hashing, "_COW_PANDAS", None)


@pytest.fixture
def reads_of_x(monkeypatch, mock_shell):
    """How many times the value named ``x`` is hashed in full."""
    seen = []
    real = content_hashers.hash_pandas

    def counting(value, *a, **k):
        if value is mock_shell.user_ns.get("x"):
            seen.append(1)
        return real(value, *a, **k)

    monkeypatch.setattr(content_hashers, "hash_pandas", counting)
    return seen


@pytest.fixture
def settled(monkeypatch):
    """What each `ArgFingerprints.settle` answered: True for unchanged."""
    answers = []
    real = call_effects.ArgFingerprints.settle

    def spy(self, value, token, digest):
        answer = real(self, value, token, digest)
        answers.append(answer == token)
        return answer

    monkeypatch.setattr(call_effects.ArgFingerprints, "settle", spy)
    return answers


SETUP = (
    "import time\n"
    "import numpy as np, pandas as pd\n"
    "box = [None]\n"
    "def summary(s):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return len(s)\n"
    "def rewrite(s):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    box[0].loc[0, 'a'] = box[0].loc[0, 'a']\n"
    "    return len(s)\n"
    "def bump(s):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    box[0].loc[0, 'a'] = box[0].loc[0, 'a'] + 1\n"
    "    return len(s)\n"
)

FRAME = "x = pd.DataFrame({'a': np.arange(300_000.0), 'b': np.ones(300_000)})\nbox[0] = x"


def _run(cash_magics, mock_shell):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, FRAME)


def test_a_frame_left_as_it_was_is_not_read(cash_magics, mock_shell, reads_of_x):
    _run(cash_magics, mock_shell)
    # As ``chg`` on p18, which nothing had hashed before ``chg_tab``: what
    # its own statement took is forgotten.
    arg_hashing.FRAME_DIGESTS.clear()
    reads_of_x.clear()
    run_cash_cell(cash_magics, "t = summary(x[x.b > 0])")
    assert mock_shell.user_ns["t"] == 300_000
    # Was 1: the statement's fingerprint before it ran (the one after it
    # found the digest that one left).
    assert len(reads_of_x) == 0, f"x was hashed {len(reads_of_x)} times"


def test_a_frame_written_with_the_same_values_is_compared_by_content(cash_magics, mock_shell, reads_of_x, settled):
    _run(cash_magics, mock_shell)
    run_cash_cell(cash_magics, "t = rewrite(x[x.b > 0])")
    # Its blocks changed, its content did not: the copy held is read
    # against the frame, and the statement changed nothing.
    assert settled == [True]
    assert reads_of_x, "the frame was not read against the copy held"


def test_a_frame_changed_reads_as_changed(cash_magics, mock_shell, settled):
    _run(cash_magics, mock_shell)
    run_cash_cell(cash_magics, "t = bump(x[x.b > 0])")
    assert settled == [False]
    assert mock_shell.user_ns["x"].loc[0, "a"] == 1.0


def test_a_frame_left_as_it_was_is_settled_unchanged(cash_magics, mock_shell, settled):
    _run(cash_magics, mock_shell)
    run_cash_cell(cash_magics, "t = summary(x[x.b > 0])")
    assert settled == [True]


def test_a_frame_handed_on_whole_keeps_its_digest_for_the_call(cash_magics, mock_shell, settled):
    """Positive control: ``summary(x)`` hands ``x`` itself, whose digest the
    call takes over, so nothing is deferred."""
    _run(cash_magics, mock_shell)
    run_cash_cell(cash_magics, "t = summary(x)")
    assert settled == []

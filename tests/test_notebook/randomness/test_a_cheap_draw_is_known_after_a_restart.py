"""A draw too cheap to store still moves its generator after a restart.

``x0 = rng.normal()`` is never written to the cache, so after a restart no
entry says it drew, and the namespace has no ``rng`` to look at. The
simulation then left ``rng`` where the fresh generator has it: a cell below
run alone rebuilt ``rng`` without the draw and drew from the wrong position.

Two things say it drew: the record the run keeps for a later kernel
(``carrier_advances_key``), and, when there is none, a generator the
simulation saw made (``default_rng(...)``). Each is pinned on its own: a
generator made by a helper of the notebook's is not recognised by its code,
so only the record can say; an in-memory cache keeps no record, so only the
code can.
"""

from __future__ import annotations

import numpy as np
import pytest

from cash.backends.file_backend import FileBackend
from cash.notebook.cache_key import statement_source_hash
from tests._cell_driver import run_cash_cell

IMPORT = "import numpy as np"
SEED = "rng = np.random.default_rng(42)"
HELPER = "def make_rng(seed):\n    return np.random.default_rng(seed)"
HELPER_SEED = "rng = make_rng(42)"
DRAW = "x0 = float(rng.normal())"
LOOK = "kind = type(rng).__name__"
BELOW = "z = float(rng.normal())"


def _oracle() -> float:
    r = np.random.default_rng(42)
    r.normal()
    return float(r.normal())


def _restart(magics, names) -> None:
    """What a kernel restart leaves: no session state, none of *names*."""
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in names:
        magics.shell.user_ns.pop(name, None)


def _codes(magics) -> list[str]:
    return [m.get("code") for m in magics.cash_status("dict")["last_cell"]["statements"] if m.get("code")]


def _run_below_alone_after_a_restart(magics, cells) -> None:
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)
    assert magics.shell.user_ns["z"] == _oracle(), "control: the top-to-bottom run"
    _restart(magics, ["np", "rng", "make_rng", "x0", "z"])
    run_cash_cell(magics, BELOW, cells=cells)
    assert DRAW in _codes(magics), f"the rebuild skipped the draw above: {_codes(magics)}"
    assert magics.shell.user_ns["z"] == _oracle()


def test_a_generator_made_by_numpy_is_recognised_without_a_record(cash_magics):
    """In memory: the restart leaves no record, only the code that made ``rng``."""
    _run_below_alone_after_a_restart(cash_magics, [IMPORT, SEED, DRAW, BELOW])


@pytest.fixture
def clean_backend(tmp_path):
    """A file backend in place of the in-memory one: the record a later
    kernel reads is metadata alone, which only a file tier keeps."""
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    yield backend
    backend.clear()


def test_the_record_says_which_generators_a_cheap_statement_drew_from(cash_magics, clean_backend):
    from cash.notebook.cache_key import carrier_advances_key

    cells = [IMPORT, SEED, DRAW, LOOK]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    drew = clean_backend.get_metadata(carrier_advances_key(statement_source_hash(DRAW)))
    looked = clean_backend.get_metadata(carrier_advances_key(statement_source_hash(LOOK)))
    assert drew is not None and drew["names"] == ["rng"], drew
    # Reading a generator without drawing is recorded too, as nothing drawn.
    assert looked is not None and looked["names"] == [], looked


def test_the_record_says_it_drew_from_a_generator_made_by_a_helper(cash_magics, clean_backend):
    """``make_rng`` hides the generator from the code: only the record says."""
    _run_below_alone_after_a_restart(cash_magics, [IMPORT, HELPER, HELPER_SEED, DRAW, BELOW])

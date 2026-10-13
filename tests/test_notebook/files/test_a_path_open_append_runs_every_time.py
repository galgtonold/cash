"""A helper of yours appending through ``Path.open('a')`` runs every time, as
``open(path, 'a')`` in the same helper does and as a plain kernel does.

The runtime observer left an append to the static check unless the user's
own code opened the file, and it walked past ``gzip`` and ``codecs`` to find
who did, but not past ``pathlib``: ``LOG.open('a')`` was a library's append,
the static check does not read ``.open`` on a path either, and the statement
was stored. On a hit the line was not written, so a log plain Python grows on
every run stopped growing.
"""

from __future__ import annotations

import gzip

import pytest

from cash.notebook.write_observer import kept_writes, observe_writes
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


def _append_via_path(path):
    with path.open("a", encoding="utf-8") as f:
        f.write("a\n")


def test_the_observer_keeps_a_path_open_append_of_the_users_code(tmp_path):
    log = tmp_path / "log.txt"
    with observe_writes() as written:
        _append_via_path(log)
    assert kept_writes(written, None, written.appended) == [str(log)]


#: Every spelling of an append a helper can make through a library's opener,
#: next to the ``open(p, 'a')`` they must behave like.
APPENDS = {
    "open": "with open(P, 'a') as f:\n        f.write('a\\n')",
    "Path.open": "with Path(P).open('a') as f:\n        f.write('a\\n')",
    "Path.open-mode-keyword": "with Path(P).open(mode='a') as f:\n        f.write('a\\n')",
    "io.open": "with io.open(P, 'a') as f:\n        f.write('a\\n')",
    "codecs.open": "with codecs.open(P, 'a', 'utf-8') as f:\n        f.write('a\\n')",
    "gzip.open": "with gzip.open(P, 'at') as f:\n        f.write('a\\n')",
}


@pytest.mark.parametrize("body", list(APPENDS.values()), ids=list(APPENDS))
def test_the_log_grows_on_every_run(cash_magics, tmp_path, body):
    log = tmp_path / "run.log"
    cells = [
        "import codecs, gzip, io, time\nfrom pathlib import Path\n"
        f"P = {str(log)!r}\n"
        "def write_log(x):\n"
        f"    {body}\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    return x",
        "n = write_log(1)",
    ]
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    for _ in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    opener = gzip.open if "gzip" in body else open
    with opener(log, "rt", encoding="utf-8") as f:
        assert f.read().split() == ["a"] * 3

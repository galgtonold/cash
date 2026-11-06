"""A structure that ends right after a share check does not ask it again.

``if not os.path.isdir(d): with tarfile.open(t) as tar: tar.extractall()``
ran the share check of ``tar`` three times with nothing run between: for the
``with`` run as one statement, for that statement's structure, and for the
``if`` around it. Each walked the 20,000 members of the ``TarFile`` (0.15 s
each on Linux, 0.8 s each on Windows for 100,000).

What must hold: once a statement started after the check, the structure
checks again.
"""

from __future__ import annotations

import io
import tarfile

import pytest

from cash.notebook import shared_objects
from tests._cell_driver import run_cash_cell


def _blob(n: int = 200) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for i in range(n):
            info = tarfile.TarInfo(f"f{i}.txt")
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    return buf.getvalue()


@pytest.fixture
def tar_walks(cash_magics, monkeypatch):
    """How many share checks walked a ``TarFile``."""
    walks = []
    real = shared_objects._held_from_outside

    def counting(nodes, owner, value_types, leaves=None):
        if any(type(node) is tarfile.TarFile for node in nodes.values()):
            walks.append(1)
        return real(nodes, owner, value_types, leaves)

    monkeypatch.setattr(shared_objects, "_held_from_outside", counting)
    cash_magics.shell.user_ns.update(blob=_blob(), io=io, tarfile=tarfile)
    return walks


ONE_UNIT = """if True:
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        names = tar.getnames()"""


def test_the_with_and_the_if_around_it_take_its_statement_s_check(cash_magics, tar_walks):
    run_cash_cell(cash_magics, ONE_UNIT)
    assert len(cash_magics.shell.user_ns["names"]) == 200
    assert len(tar_walks) == 1


def test_a_statement_after_it_in_the_branch_has_the_if_check_again(cash_magics, tar_walks):
    run_cash_cell(cash_magics, ONE_UNIT + "\n    n = len(names)")
    assert cash_magics.shell.user_ns["n"] == 200
    assert len(tar_walks) == 2, "control: the if's own check ran after another statement"

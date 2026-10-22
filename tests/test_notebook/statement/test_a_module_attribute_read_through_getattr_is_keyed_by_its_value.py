"""``getattr(mylib, 'val_a')`` reads ``mylib.val_a`` as the plain spelling
does, and ``getattr(mylib, 'val_' + name)`` may read any of the module's
names: both are keyed by what the local module holds, so a change made to
it outside the notebook's cells (the console, a thread) reaches the
statement and the reader below it instead of serving them the old value.
"""

from __future__ import annotations

import pytest

from cash.backends.persistence_policy import PersistencePolicy
from tests._cell_driver import run_cash_cell

#: How the statement reads the module's ``val_a``.
READS = {
    "a plain attribute": "v = mylib.val_a",
    "getattr with the name spelled": "v = getattr(mylib, 'val_a')",
    "getattr with a name made when it runs": "v = getattr(mylib, 'val_' + name)",
    "getattr with a default": "v = getattr(mylib, 'val_' + name, None)",
    "getattr in a function": "def get(n):\n    return getattr(mylib, 'val_' + n)\nv = get(name)",
}


@pytest.mark.parametrize("how", list(READS))
def test_a_change_to_the_module_reaches_the_reader(cash_magics, monkeypatch, tmp_path, how):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    (tmp_path / "mylib_getattr_probe.py").write_text("val_a = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(__import__("sys").modules, "mylib_getattr_probe", raising=False)
    cells = ["import mylib_getattr_probe as mylib\nname = 'a'", READS[how], "w = [v, 'read']"]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert ns["w"] == [1, "read"]

    ns["mylib"].val_a = 2
    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert ns["w"] == [2, "read"], "the reader was served the old value"

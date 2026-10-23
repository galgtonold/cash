"""``getattr(mylib, 'val_a')`` reads ``mylib.val_a`` as the plain spelling
does, and ``getattr(mylib, 'val_' + name)`` may read any of the module's
names: both are keyed by what the local module holds, so a change made to
it outside the notebook's cells (the console, a thread) reaches the
statement and the reader below it instead of serving them the old value.
So is a module used whole in any other way: ``vars(mylib)``, its
``__dict__``, or handed to a function that reads it by name.

A function that calls ``getattr`` on something else is keyed only by the
module names it spells.
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
    "getattr on a submodule": "v = getattr(mylib.sub, 'val_' + name)",
    "getattr on a module handed to a function": "def get(mod, n):\n    return getattr(mod, 'val_' + n)\nv = get(mylib, name)",
    "getattr on an alias in a function": "def get(n):\n    m = mylib\n    return getattr(m, 'val_' + n)\nv = get(name)",
    "vars of the module": "v = vars(mylib)['val_' + name]",
    "the module's __dict__": "v = mylib.__dict__['val_' + name]",
    "the module's __dict__ in a function": "def get(n):\n    return mylib.__dict__['val_' + n]\nv = get(name)",
    "operator.attrgetter": "import operator\nv = operator.attrgetter('val_' + name)(mylib)",
}
#: What changes for each read, when not ``mylib.val_a``.
CHANGES = {"getattr on a submodule": "mylib.sub.val_a = 2"}


def _modules(monkeypatch, tmp_path, **files):
    for name, text in files.items():
        (tmp_path / f"{name}.py").write_text(text)
        monkeypatch.delitem(__import__("sys").modules, name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))


@pytest.mark.parametrize("how", list(READS))
def test_a_change_to_the_module_reaches_the_reader(cash_magics, monkeypatch, tmp_path, how):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    _modules(
        monkeypatch,
        tmp_path,
        mylib_getattr_probe="val_a = 1\nimport mylib_getattr_probe_sub as sub\n",
        mylib_getattr_probe_sub="val_a = 1\n",
    )
    cells = ["import mylib_getattr_probe as mylib\nname = 'a'", READS[how], "w = [v, 'read']"]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert ns["w"] == [1, "read"]

    exec(CHANGES.get(how, "mylib.val_a = 2"), ns)
    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert ns["w"] == [2, "read"], "the reader was served the old value"


@pytest.mark.parametrize("body", ["o.real", "getattr(o, 'real')"])
def test_a_change_to_a_name_a_function_never_reads_is_served(cash_magics, monkeypatch, tmp_path, body):
    monkeypatch.setattr(PersistencePolicy, "too_cheap_to_store", lambda self, compute_s: False)
    _modules(monkeypatch, tmp_path, wlib_getattr_probe="K = 1\nJ = 2\nBIG = list(range(300_000))\n")
    cells = ["import wlib_getattr_probe as wlib", f"def usek(o):\n    return {body} + wlib.K", "v = usek(3)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    exec("wlib.J = 99", cash_magics.shell.user_ns)
    run_cash_cell(cash_magics, cells[2], cells=cells)
    (statement,) = cash_magics.cash_status("dict")["last_cell"]["statements"]
    assert str(statement["status"]).endswith("RESTORED")

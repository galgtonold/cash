"""A statement whose function imports a module by a string its trace cannot
follow runs every time, and says so.

``m = importlib.import_module(name)`` or ``__import__('wmod')`` in a helper
picks the module at run time: the key cannot name it, so the statement runs
every time, which is right, but no warning said why, while the same helper
importing ``from wmod import scale`` is served.
"""

import warnings

import pytest

from tests._cell_driver import run_cash_cell

BODIES = {
    "import_module_by_a_name": "import importlib\n    m = importlib.import_module(name)\n    scale = m.scale",
    "import_module_by_a_constant": "import importlib\n    scale = importlib.import_module('wmod').scale",
    "dunder_import": "scale = __import__('wmod').scale",
}


@pytest.mark.parametrize("body", list(BODIES))
def test_it_runs_every_time_with_a_warning(cash_magics, tmp_path, monkeypatch, body):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(__import__("sys").modules, "wmod", raising=False)
    (tmp_path / "wmod.py").write_text("def scale(x):\n    return x * 2\n", encoding="utf-8")
    cells = [
        f"import time\ndef g(vals, name='wmod'):\n    time.sleep(0.05)\n    {BODIES[body]}\n    return sum(scale(v) for v in vals)",
        "y = g([1, 2, 3])",
    ]
    said = []
    for _ in range(2):
        for cell in cells:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                run_cash_cell(cash_magics, cell, cells=cells)
            said += [w for w in caught if getattr(w.message, "code", None) == "KEY-UNTRACKABLE-DEP"]
    assert cash_magics.shell.user_ns["y"] == 12
    assert len(said) == 1, [str(w.message) for w in said]
    assert "importlib.import_module" in str(said[0].message)

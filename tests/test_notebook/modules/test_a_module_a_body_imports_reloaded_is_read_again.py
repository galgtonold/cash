"""A module a function imports in its body, edited and reloaded, is what
the next call runs.

``def g(vals): from bodymod import scale; ...``, then ``bodymod.py``
rewritten and ``importlib.reload(sys.modules['bodymod'])``: ``z = g(...)``
was served what the old ``scale`` returned, as the import binds a local
that no lineage follows. The key now holds what the file of each module of
the user's a called function imports holds.
"""

from __future__ import annotations

import sys

import pytest

from tests._cell_driver import run_cash_cell


@pytest.fixture
def in_a_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    sys.modules.pop("bodymod", None)


@pytest.mark.parametrize(
    "body_import",
    ["from bodymod import scale", "import bodymod\n    scale = bodymod.scale"],
    ids=["from_import", "module_import"],
)
def test_the_call_after_a_reload_runs_the_new_code(cash_magics, in_a_project, body_import):
    cells = [
        "open('bodymod.py', 'w').write('def scale(x):\\n    return x * 2\\n')",
        f"import time\ndef g(vals):\n    time.sleep(0.05)\n    {body_import}\n    return sum(scale(v) for v in vals)",
        "y = g([1, 2, 3])",
        "import importlib, sys\n"
        "open('bodymod.py', 'w').write('def scale(x):\\n    return x * 30\\n')\n"
        "importlib.reload(sys.modules['bodymod'])",
        "z = g([1, 2, 3])",
    ]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert (ns["y"], ns["z"]) == (12, 180)

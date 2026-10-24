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


@pytest.fixture
def a_package(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "bodypkg").mkdir()
    (tmp_path / "bodypkg" / "__init__.py").write_text("", encoding="utf-8")
    yield
    for name in [n for n in sys.modules if n == "bodypkg" or n.startswith("bodypkg.")]:
        sys.modules.pop(name)


@pytest.mark.parametrize(
    "body_import, use",
    [("from .helper import scale", "scale(v)"), ("from . import helper", "helper.scale(v)")],
    ids=["from_a_sibling", "the_sibling_itself"],
)
def test_a_relative_import_in_a_package_function_is_read_again(cash_magics, a_package, body_import, use):
    """``from .helper import scale`` in ``bodypkg/api.py`` gave the bare
    name ``helper``, looked up as a module of its own, and ``from . import
    helper`` an empty one, dropped: neither keyed ``bodypkg/helper.py``."""
    api = f"import time\ndef g(vals):\n    time.sleep(0.05)\n    {body_import}\n    return sum({use} for v in vals)\n"
    cells = [
        f"open('bodypkg/api.py', 'w').write({api!r})\n"
        "open('bodypkg/helper.py', 'w').write('def scale(x):\\n    return x * 2\\n')",
        "from bodypkg.api import g",
        "y = g([1, 2, 3])",
        "import importlib, sys\n"
        "open('bodypkg/helper.py', 'w').write('def scale(x):\\n    return x * 30\\n')\n"
        "importlib.reload(sys.modules['bodypkg.helper'])",
        "z = g([1, 2, 3])",
    ]
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert (ns["y"], ns["z"]) == (12, 180)

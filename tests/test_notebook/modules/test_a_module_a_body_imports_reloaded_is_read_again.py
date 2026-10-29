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


def _reload_cells(setup: str, fn: str, helper: str) -> list[str]:
    path = helper.replace(".", "/") + ".py"
    return [
        setup + f"\nopen({path!r}, 'w').write('def scale(x):\\n    return x * 2\\n')",
        fn,
        "y = g([1, 2, 3])",
        "import importlib, sys\n"
        f"open({path!r}, 'w').write('def scale(x):\\n    return x * 30\\n')\n"
        f"importlib.reload(sys.modules[{helper!r}])",
        "z = g([1, 2, 3])",
    ]


def test_an_import_in_a_method_the_function_calls_is_read_again(cash_magics, a_package):
    """``g`` calls ``K().run(vals)``, whose body does ``from .helper import
    scale``: the walk resolved ``K`` in the notebook, which does not hold
    it, so ``bodypkg/helper.py`` was in no key."""
    api = (
        "import time\nclass K:\n    def run(self, vals):\n        time.sleep(0.05)\n"
        "        from .helper import scale\n        return sum(scale(v) for v in vals)\n"
        "def g(vals):\n    return K().run(vals)\n"
    )
    cells = _reload_cells(f"open('bodypkg/api.py', 'w').write({api!r})", "from bodypkg.api import g", "bodypkg.helper")
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert (ns["y"], ns["z"]) == (12, 180)


def test_a_relative_import_of_a_module_gone_from_sys_modules_is_read_again(cash_magics, a_package):
    """``del sys.modules['bodypkg.api']`` after ``from bodypkg.api import
    g``: the package was read off the loaded module whose file ``g``'s code
    is in, found no more, and the relative import was dropped. It is read
    off ``g``'s own globals."""
    api = "import time\ndef g(vals):\n    time.sleep(0.05)\n    from .helper import scale\n    return sum(scale(v) for v in vals)\n"
    cells = _reload_cells(
        f"open('bodypkg/api.py', 'w').write({api!r})",
        "import sys\nfrom bodypkg.api import g\ndel sys.modules['bodypkg.api']",
        "bodypkg.helper",
    )
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert (ns["y"], ns["z"]) == (12, 180)


def test_a_package_function_importing_its_submodule_does_not_rewrite_itself(cash_magics, a_package):
    """``from .helper import scale`` in a function of ``bodypkg/__init__.py``
    sets ``bodypkg.helper`` as the import system does. Taken for state the
    call set, ``y = g(...)`` became the writer of ``g``, moved its lineage,
    and a Run All after a restart ran it again below the reload of
    ``helper``: ``y`` was 180 where plain Python gives 12."""
    init = "import time\ndef g(vals):\n    time.sleep(0.05)\n    from .helper import scale\n    return sum(scale(v) for v in vals)\n"
    cells = _reload_cells(
        f"open('bodypkg/__init__.py', 'w').write({init!r})", "from bodypkg import g", "bodypkg.helper"
    )
    ns = cash_magics.shell.user_ns
    lineage = cash_magics.tracking_state.variable_lineage
    for cell in cells[:2]:
        run_cash_cell(cash_magics, cell, cells=cells)
    before = lineage["g"]
    run_cash_cell(cash_magics, cells[2], cells=cells)
    assert lineage["g"] == before
    for cell in cells[3:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert (ns["y"], ns["z"]) == (12, 180)


def test_a_hit_after_a_restart_imports_what_the_call_imports(cash_magics, a_package):
    """``y = g(...)`` served after a restart did not run ``from .helper
    import scale``, so ``bodypkg.helper`` was never imported and
    ``importlib.reload(sys.modules['bodypkg.helper'])`` below raised the
    ``KeyError`` plain Python does not."""
    api = "import time\ndef g(vals):\n    time.sleep(0.3)\n    from .helper import scale\n    return sum(scale(v) for v in vals)\n"
    cells = _reload_cells(f"open('bodypkg/api.py', 'w').write({api!r})", "from bodypkg.api import g", "bodypkg.helper")
    ns = cash_magics.shell.user_ns
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    for name in [n for n in sys.modules if n.startswith("bodypkg.")]:
        del sys.modules[name]
    cash_magics.tracking_state.reset_session_state()
    for name in ("g", "y", "z"):
        ns.pop(name)
    statuses = []
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
        statuses.append([str(s.get("status")) for s in cash_magics._last_cell_metrics.get("statements", [])])
    assert "RESTORED" in statuses[2], statuses
    assert (ns["y"], ns["z"]) == (12, 180)

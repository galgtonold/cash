"""A statement that leaves a ``weakref.finalize`` waiting runs again.

``weakref.finalize(cfg['ns'], print, 'gone')`` registers the finalizer in
``weakref.finalize``'s own registry, which no variable holds: restored from
the cache, the statement registered none, whoever's the object and the
callback. ``h.d = tempfile.TemporaryDirectory()`` restored named a
directory the old object's finalizer had already deleted. A finalizer a
module makes while it is first imported is the module's, made once a
process whoever imports it (torch makes 44 on an optimizer's first step):
it keeps nothing out.
"""

from __future__ import annotations

import gc
import sys
import textwrap

import pytest

from tests._cell_driver import run_cash_cell


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    gc.collect()


HOLDER = "import argparse, functools, weakref\nclass H:\n    pass\nh = H()\nh.ns = argparse.Namespace(a=1)"
#: (setup cell, the cell making the finalizer, the object it watches)
CASES = {
    "an-item-of-a-dict": (
        "import argparse, weakref\ncfg = {'ns': argparse.Namespace(a=1)}",
        "weakref.finalize(cfg['ns'], print, 'gone')",
        "cfg['ns']",
    ),
    "an-attribute": (HOLDER, "weakref.finalize(h.ns, print, 'gone')", "h.ns"),
    "a-partial-callback": (HOLDER, "weakref.finalize(h.ns, functools.partial(print, 'gone'))", "h.ns"),
    "bound-to-a-name": (HOLDER, "f = weakref.finalize(h.ns, print, 'gone')", "h.ns"),
    "an-item-of-a-list": (
        "import argparse, weakref\nitems = [argparse.Namespace(a=1)]",
        "weakref.finalize(items[0], print, 'gone')",
        "items[0]",
    ),
    "in-a-loop": (
        "import argparse, weakref\nclass H:\n    pass\nhs = [H() for _ in range(2)]\n"
        "for x in hs:\n    x.ns = argparse.Namespace(a=1)",
        "for x in hs:\n    weakref.finalize(x.ns, print, 'gone')",
        "hs[0].ns",
    ),
    "a-temporary-directory": (
        "import os, tempfile, weakref\nclass H:\n    pass\nh = H()",
        "h.d = tempfile.TemporaryDirectory()",
        "h.d",
    ),
}


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize("kind", list(CASES))
def test_run_all_again_leaves_the_finalizer_plain_python_does(kind, cash_magics, mock_shell):
    setup, make, watched = CASES[kind]
    probe = f"nfin = sum(1 for _i in list(weakref.finalize._registry.values()) if _i.weakref() is {watched})"
    if kind == "a-temporary-directory":
        probe += "\nthere = os.path.isdir(h.d.name)"
    cells = [setup, make, probe]
    try:
        for _ in range(2):  # Run All, twice
            for cell in cells[:2]:
                run_cash_cell(cash_magics, cell, cells=cells)
        statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
        run_cash_cell(cash_magics, probe, cells=cells)
        ns = mock_shell.user_ns
        assert ns["nfin"] == 1, f"plain Python leaves one finalizer on {watched}: {statuses}"
        if kind == "a-temporary-directory":
            assert ns["there"], f"the directory was deleted: {statuses}"
    finally:
        gc.collect()


def _first_use_twice(cash_magics, tmp_path, monkeypatch, name: str, body: str) -> list:
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(body))
    monkeypatch.delitem(sys.modules, name, raising=False)
    setup = f"def first_use_{name}(n):\n    import {name}\n    return n + 1"
    cells = [setup, f"y = first_use_{name}(1)"]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    return [(str(s["status"]), s.get("uncacheable_reasons")) for s in cash_magics._last_cell_metrics["statements"]]


@pytest.mark.usefixtures("store_everything")
def test_a_finalizer_a_module_makes_on_its_first_import_keeps_nothing_out(cash_magics, tmp_path, monkeypatch):
    """As the same statement over a module that makes none."""
    monkeypatch.syspath_prepend(str(tmp_path))
    plain = _first_use_twice(cash_magics, tmp_path, monkeypatch, "zz_plain_module", "VALUE = 1\n")
    finalizing = _first_use_twice(
        cash_magics,
        tmp_path,
        monkeypatch,
        "zz_finalizing_module",
        """
        import weakref

        class Resource:
            pass

        RESOURCE = Resource()
        weakref.finalize(RESOURCE, print, "module resource gone")
        """,
    )
    assert finalizing == plain

"""A cached function that calls a method of a module-level instance
(``model = Model()``, ``model.run(x)``) keeps that method found by its
report after the walk, and a report read back from disk is the one walking
again gives.

The walk visits the method twice: as its function (``Model.run``) and as
the method bound to ``model`` (``Model.run#2``). The bound method is made
anew on every attribute read, and the report held it by a plain weak
reference, which died with the walk: with ``CASH_ANALYSIS_MEMO_VERIFY=1``
the report walked again could not be spelled ("helper Model.run#2 is
gone"), and the cached function ran uncached. It is held by what it binds
(`helper_bindings.held_ref`), as the walk's call-site bindings are.

The key never depended on the dead reference -- the function itself is
keyed under its own name -- so an edit to ``Model.run`` moves the key in a
fresh process, with the report stored or walked again.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from tests._scripts import run_python

_RUNNER = textwrap.dedent(
    """
    import gc, json, sys, warnings
    sys.path.insert(0, sys.argv[1])
    import cash, app
    from cash.analysis import report_store
    from cash.analysis.helper_walk import HelperWalk

    report = HelperWalk(app.top).run()
    gc.collect()
    walked_gone = sorted(q for q, ref in report.helper_objects.items() if ref() is None)
    loaded = report_store.load(app.top)
    loaded_gone = None if loaded is None else sorted(q for q, ref in loaded.helper_objects.items() if ref() is None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        top = cash.cache(app.top)
        value = top(3)
        key = top.explain(3).cache_key
    report_store.flush()
    print(json.dumps({
        "stats": report_store.status(),
        "key": key,
        "value": value,
        "names": sorted(report.helper_source_hashes),
        "walked_gone": walked_gone,
        "loaded_gone": loaded_gone,
        "warnings": [str(w.message) for w in caught],
    }))
    """
)

# `top` reaches `model.run` directly and through `via`.
_APP = """
from models import model


def via(x):
    return model.run(x)


def top(n):
    return model.run(n) + via(n)
"""

_MODELS = """
K = 10


class Model:
    def run(self, x):
        return x + K


model = Model()
"""


def _run(code_dir: Path, verify: bool = False) -> dict:
    env = {"CASH_ANALYSIS_MEMO_VERIFY": "1" if verify else None}
    out = run_python("-c", _RUNNER, str(code_dir), cwd=code_dir, cache_dir=code_dir / ".cash", env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _write(code: Path, models: str = _MODELS) -> None:
    code.mkdir(exist_ok=True)
    (code / "models.py").write_text(models, encoding="utf-8")
    (code / "app.py").write_text(_APP, encoding="utf-8")


def test_the_bound_method_outlives_the_walk_walked_or_read_back(tmp_path):
    code = tmp_path / "code"
    _write(code)
    first = _run(code)
    assert first["value"] == 26
    # The case at hand: the method met as its function and as bound to `model`.
    assert "models.Model.run#2" in first["names"], first["names"]
    assert first["walked_gone"] == [], first["walked_gone"]
    second = _run(code)
    assert second["stats"]["loaded"] >= 1, second["stats"]
    assert second["loaded_gone"] == [], second["loaded_gone"]
    assert (second["key"], second["value"]) == (first["key"], first["value"])


def test_verify_mode_finds_the_stored_report_the_same_as_a_walk(tmp_path):
    code = tmp_path / "code"
    _write(code)
    _run(code)
    checked = _run(code, verify=True)
    assert checked["stats"]["verified"] >= 1, checked["stats"]
    assert not any("StoredReportMismatch" in w or "KEY-HELPERS-UNWALKABLE" in w for w in checked["warnings"]), checked[
        "warnings"
    ]
    assert checked["key"] is not None and checked["value"] == 26


def test_an_edit_to_the_method_moves_the_key_in_a_fresh_process(tmp_path):
    code = tmp_path / "code"
    _write(code)
    first = _run(code)
    assert _run(code)["stats"]["loaded"] >= 1
    _write(code, _MODELS.replace("return x + K", "return x + K + 1000"))
    for verify in (False, True):
        again = _run(code, verify=verify)
        assert again["value"] == 2026, again
        assert again["key"] != first["key"]
        assert not any("StoredReportMismatch" in w for w in again["warnings"]), again["warnings"]
    # The global the method reads is keyed too.
    _write(code, _MODELS.replace("K = 10", "K = 20"))
    moved = _run(code)
    assert moved["value"] == 46 and moved["key"] != first["key"]

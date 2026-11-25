"""``CASH_ANALYSIS_MEMO_VERIFY=1`` -- every report read from disk is walked
again and compared -- passes for a cached function whose helper's own name
does not lead back to it: ``A = lambda x: x + 1``, or a def bound under
another name.

A stored report spells each helper by the path the walk reached it by
(``h.A``). The check spelled the report walked again without that walk, so
such a helper had no path ("no name leads to <function <lambda>>"), and
every function calling one was reported as a `StoredReportMismatch` and ran
uncached -- the one tool that can catch a stale stored report was red for
the helpers hardest to find again. A report that really differs is still
caught.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = pytest.mark.core

_RUNNER = textwrap.dedent(
    """
    import dataclasses, json, sys, warnings
    sys.path.insert(0, sys.argv[1])
    import cash, main
    from cash.analysis import report_store
    from cash.analysis.helper_walk import HelperWalk

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f = cash.cache(main.f)
        value = f(1)
        key = f.explain(1).cache_key
    report_store.flush()
    tampered = None
    loaded = report_store.load(main.f)
    if loaded is not None:
        walk = HelperWalk(main.f)
        walked = walk.run()
        report_store.verify(main.f, loaded, walked, walk)
        other = dataclasses.replace(loaded, opaque_callees=(*loaded.opaque_callees, "somewhere.else"))
        try:
            report_store.verify(main.f, other, walked, walk)
            tampered = "passed"
        except report_store.StoredReportMismatch as e:
            tampered = str(e)
    print(json.dumps({
        "stats": report_store.status(),
        "value": value,
        "key": key,
        "tampered": tampered,
        "warnings": [str(w.message) for w in caught],
    }))
    """
)

_MAIN = "import h\n\n\ndef f(x):\n    return h.A(x)\n"
HELPERS = {
    "lambda": "A = lambda x: x + 1\n",
    "renamed def": "def _a(x):\n    return x + 1\n\n\nA = _a\ndel _a\n",
    "plain def": "def A(x):\n    return x + 1\n",
}


def _run(code, verify):
    env = {"CASH_ANALYSIS_MEMO_VERIFY": "1" if verify else None}
    out = run_python("-c", _RUNNER, str(code), cwd=code, cache_dir=code / ".cash", env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("helper", list(HELPERS))
def test_the_check_passes_when_nothing_changed(tmp_path, helper):
    code = tmp_path / "code"
    code.mkdir()
    (code / "main.py").write_text(_MAIN, encoding="utf-8")
    (code / "h.py").write_text(HELPERS[helper], encoding="utf-8")
    first = _run(code, verify=False)
    assert first["value"] == 2
    checked = _run(code, verify=True)
    # The check ran, on a report read from disk.
    assert checked["stats"]["loaded"] >= 1 and checked["stats"]["verified"] >= 1, checked["stats"]
    assert not any("StoredReportMismatch" in w or "KEY-HELPERS-UNWALKABLE" in w for w in checked["warnings"]), checked[
        "warnings"
    ]
    assert (checked["value"], checked["key"]) == (2, first["key"])
    # A report that is not what the walk gives is still a mismatch.
    assert checked["tampered"] is not None and "differs" in checked["tampered"], checked["tampered"]


@pytest.mark.parametrize("helper", ["lambda", "renamed def"])
def test_an_edit_to_the_helper_is_seen_under_the_check(tmp_path, helper):
    code = tmp_path / "code"
    code.mkdir()
    (code / "main.py").write_text(_MAIN, encoding="utf-8")
    (code / "h.py").write_text(HELPERS[helper], encoding="utf-8")
    first = _run(code, verify=False)
    (code / "h.py").write_text(HELPERS[helper].replace("x + 1", "x + 100"), encoding="utf-8")
    edited = _run(code, verify=True)
    assert edited["value"] == 101 and edited["key"] != first["key"]
    assert not any("StoredReportMismatch" in w for w in edited["warnings"]), edited["warnings"]

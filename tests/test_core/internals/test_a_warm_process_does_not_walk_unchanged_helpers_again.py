"""A fresh process reads the purity report of a cached function an earlier
process walked, while everything the walk read is the same, and walks again
when any of it moved.

Every new process walked each helper a cached function reaches before its
first call: read, parsed and audited every body, though the code had not
changed since the last run. The report is stored with what it was worked out
from (`cash.analysis.report_store`): the text of every function the walk
visited and of the files they live in, what their names are bound to, the
call-site bindings. An unchanged program walks nothing; an edit to any
helper the function reaches -- one called through another module, a comment
that waives a finding, a global of its file -- or a name bound to another
function walks again, and the key and the findings are the ones a walk
without the store gives.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from tests._scripts import run_python

_RUNNER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import cash, app
    if sys.argv[2:] == ["rebind"]:
        app.step = app.other_step
    from cash.analysis import helper_walk, report_store
    for arg in sys.argv[2:]:
        if arg.startswith("fail:"):
            def failing(*args, **kwargs):
                raise NameError("a store that broke")
            setattr(report_store, arg[5:], failing)
    walked = []
    plain = helper_walk.HelperWalk._walk_one
    def counted(self, entry, qualname):
        walked.append(qualname)
        return plain(self, entry, qualname)
    helper_walk.HelperWalk._walk_one = counted
    top = cash.cache(app.top)
    key = top.explain(3).cache_key
    value = top(3)
    from cash.analysis.purity_analyzer import get_analyzer
    issues = sorted(f"{i.kind}: {i.description}" for i in get_analyzer().analyze(app.top).issues)
    report_store.flush()
    print(json.dumps({"stats": report_store.status(), "walked": walked, "key": key, "value": value,
                      "issues": issues}))
    """
)

_APP = """
import lib

LOG = []


def other_step(x):
    return x * 10


def step(x):
    return lib.leaf(x) + 1


def top(n):
    LOG.append(n)
    return step(n) * 2
"""

_LIB = """
FACTOR = 3


def leaf(x):
    return x * FACTOR
"""


def _run(code_dir: Path, *args: str, store: bool = True, verify: bool = False) -> dict:
    env = {"CASH_ANALYSIS_STORE": None if store else "0", "CASH_ANALYSIS_MEMO_VERIFY": "1" if verify else None}
    out = run_python("-c", _RUNNER, str(code_dir), *args, cwd=code_dir, cache_dir=code_dir / ".cash", env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture
def code_dir(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "app.py").write_text(_APP, encoding="utf-8")
    (code / "lib.py").write_text(_LIB, encoding="utf-8")
    return code


def _same_as_without_the_store(code_dir: Path, result: dict, *args: str) -> None:
    plain = _run(code_dir, *args, store=False)
    assert plain["stats"]["loaded"] == plain["stats"]["stored"] == 0
    assert (result["key"], result["issues"], result["value"]) == (plain["key"], plain["issues"], plain["value"])


def test_an_unchanged_program_walks_no_helper_in_a_new_process(code_dir):
    first = _run(code_dir)
    assert "lib.leaf" in first["walked"] and first["stats"]["stored"] >= 1
    assert first["issues"]  # LOG.append(n) changes a global
    second = _run(code_dir)
    assert second["walked"] == []
    assert second["stats"]["loaded"] >= 1
    assert (second["key"], second["issues"], second["value"]) == (first["key"], first["issues"], first["value"])
    # The verify mode walks again and compares: the stored report is the walked one.
    checked = _run(code_dir, verify=True)
    assert checked["stats"]["verified"] >= 1
    _same_as_without_the_store(code_dir, second)


@pytest.mark.parametrize(
    "file, old, new",
    [
        ("lib.py", "return x * FACTOR", "return x * FACTOR + 1"),  # a helper reached through another module
        ("lib.py", "FACTOR = 3", "FACTOR = 4"),  # a global of a helper's file
        ("app.py", "LOG.append(n)", "LOG.append(n)  # @cash:assume-safe"),  # a waiver: a comment
    ],
    ids=["helper-two-calls-down", "global-of-its-file", "waiver-comment"],
)
def test_an_edit_to_what_the_walk_read_walks_again(code_dir, file, old, new):
    before = _run(code_dir)
    assert _run(code_dir)["walked"] == []
    path = code_dir / file
    path.write_text(path.read_text("utf-8").replace(old, new), encoding="utf-8")
    after = _run(code_dir, verify=True)
    assert after["walked"], "the edited program was not walked again"
    assert after["stats"]["loaded"] == 0
    if file == "lib.py" and "FACTOR + 1" in new:
        assert after["key"] != before["key"]
    if "assume-safe" in new:
        assert after["issues"] != before["issues"]
    _same_as_without_the_store(code_dir, after)
    again = _run(code_dir)
    assert again["walked"] == []
    assert (again["key"], again["issues"]) == (after["key"], after["issues"])


def test_a_name_bound_to_another_function_walks_again(code_dir):
    first = _run(code_dir)
    rebound = _run(code_dir, "rebind", verify=True)
    assert "app.other_step" in rebound["walked"]
    assert rebound["key"] != first["key"]
    _same_as_without_the_store(code_dir, rebound, "rebind")
    back = _run(code_dir)
    assert back["walked"] == []
    assert back["key"] == first["key"]


def test_a_damaged_store_file_is_walked_again(code_dir):
    first = _run(code_dir)
    files = [p for p in (code_dir / ".cash" / "analysis").rglob("*.json") if "purity-report" in p.read_text("utf-8")]
    assert files
    for file in files:
        data = json.loads(file.read_text("utf-8"))
        for entry in data["entries"]:
            entry["paths"] = [["app", ["nowhere"], ""]]
        file.write_text(json.dumps(data), encoding="utf-8")
    again = _run(code_dir)
    assert again["walked"]
    assert (again["key"], again["issues"]) == (first["key"], first["issues"])
    for file in files:
        file.write_text("\x00 not json", encoding="utf-8")
    assert _run(code_dir)["walked"]


@pytest.mark.parametrize("broken", ["load", "_check", "save", "_object_guard"])
def test_a_store_that_fails_walks_and_still_sees_an_edit(code_dir, broken):
    """Whatever goes wrong reading or storing a report, the function is
    walked: never keyed without its helpers, nor run uncached."""
    plain = _run(code_dir, store=False)
    if broken in ("load", "_check"):
        _run(code_dir)  # a stored report to fail reading
    failed = _run(code_dir, f"fail:{broken}")
    again = _run(code_dir, f"fail:{broken}")
    for result in (failed, again):
        assert "lib.leaf" in result["walked"], "a store that failed stood for the walk"
        assert result["stats"]["loaded"] == 0
        assert (result["key"], result["issues"], result["value"]) == (plain["key"], plain["issues"], plain["value"])
    path = code_dir / "lib.py"
    path.write_text(path.read_text("utf-8").replace("return x * FACTOR", "return x * FACTOR + 1"), encoding="utf-8")
    edited = _run(code_dir, f"fail:{broken}")
    assert "lib.leaf" in edited["walked"]
    assert edited["key"] != plain["key"]
    _same_as_without_the_store(code_dir, edited)


def test_a_store_file_nested_too_deep_to_read_is_walked_and_written_again(code_dir):
    first = _run(code_dir)
    files = [p for p in (code_dir / ".cash" / "analysis").rglob("*.json") if "purity-report" in p.read_text("utf-8")]
    assert files
    deep = "[" * 200_000 + "]" * 200_000
    for file in files:
        file.write_text(deep, encoding="utf-8")
    again = _run(code_dir)
    assert again["walked"]
    assert (again["key"], again["issues"], again["value"]) == (first["key"], first["issues"], first["value"])
    for file in files:
        assert json.loads(file.read_text("utf-8"))["kind"] == "purity-report"
    assert _run(code_dir)["walked"] == []

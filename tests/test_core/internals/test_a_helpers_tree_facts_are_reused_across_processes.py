"""A fresh process reuses the facts earlier processes read off each helper's
tree, and only for the same text.

Every new process walked each helper again, and most of that walk was the
fresh-value flow of each body and of every user helper a body calls
(`purity_flow.returns_fresh`): which calls build a new object, which names
hold one where they are used, what each function returns. Those facts
follow from a function's source text alone, given which of its calls build
a new object; what the names are bound to is still resolved live on every
walk, and only that outcome is part of what a fact is stored under
(`cash.analysis.fact_store`). So an unchanged helper works out nothing
again, an edited one (and a caller whose calls it changes) does, a global
bound to another function is seen, and the key and the findings are the
same with the store as without.
"""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

import pytest

from tests._scripts import run_python

_RUNNER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import cash, mod
    if sys.argv[2:] == ["rebind"]:
        mod.make = mod.make_shared
    from cash.analysis import fact_store
    from cash.analysis.helper_walk import HelperWalk
    report = HelperWalk(mod.top).run()
    key = cash.cache(mod.top).explain(3).cache_key
    issues = sorted(f"{i.kind}: {i.description}" for i in report.issues)
    print(json.dumps({"stats": fact_store.status(), "issues": issues, "key": key}))
    """
)

_MODULE = """
import numpy as np

SHARED = np.zeros(3)


def make_fresh(n):
    out = np.zeros(n)
    return out


def make_shared(n):
    return SHARED


make = make_fresh


def top(n):
    v = make(n)
    v[0] = 1.0
    return float(v.sum())
"""


def _run(code_dir: Path, *args: str, store: bool = True, verify: bool = False) -> dict:
    env = {"CASH_ANALYSIS_STORE": None if store else "0", "CASH_ANALYSIS_MEMO_VERIFY": "1" if verify else None}
    out = run_python("-c", _RUNNER, str(code_dir), *args, cwd=code_dir, cache_dir=code_dir / ".cash", env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture
def code_dir(tmp_path):
    pytest.importorskip("numpy")
    code = tmp_path / "code"
    code.mkdir()
    (code / "mod.py").write_text(_MODULE, encoding="utf-8")
    (code / ".cash").mkdir()  # the store lives in an existing cache dir
    return code


def _store_files(code_dir: Path) -> list[Path]:
    return [p for p in (code_dir / ".cash" / "analysis").rglob("*.json") if "tree-facts" in p.read_text("utf-8")]


def test_a_second_process_works_out_nothing_again_for_unchanged_code(code_dir):
    first = _run(code_dir)
    assert first["stats"]["computed"] > 0 and first["stats"]["stored"] >= 1
    second = _run(code_dir, verify=True)
    assert second["stats"]["computed"] == 0
    assert second["stats"]["loaded"] == first["stats"]["computed"]
    assert second["stats"]["verified"] == second["stats"]["loaded"]
    assert second["key"] == first["key"]
    assert second["issues"] == first["issues"] == []


def test_the_key_and_findings_are_the_same_without_the_store(code_dir):
    _run(code_dir)
    stored = _run(code_dir)
    assert stored["stats"]["loaded"] > 0
    plain = _run(code_dir, store=False)
    assert plain["stats"] == {"loaded": 0, "computed": 0, "stored": 0, "refused": 0, "verified": 0}
    assert (stored["key"], stored["issues"]) == (plain["key"], plain["issues"])


def test_an_edited_helper_is_worked_out_again_with_its_callers(code_dir):
    before = _run(code_dir)
    mod = code_dir / "mod.py"
    # The helper now hands back an array a global holds: writing into it
    # is a change outside the cached function.
    mod.write_text(_MODULE.replace("    out = np.zeros(n)\n    return out", "    return SHARED"), encoding="utf-8")
    after = _run(code_dir, verify=True)
    assert after["stats"]["computed"] > 0  # the helper's facts, and the caller's flow
    assert after["stats"]["loaded"] > 0  # what did not change: make_shared, the caller's calls
    assert after["issues"] == ["scope_mutation: v[...] = ... - subscript mutation"]
    assert after["key"] != before["key"]
    plain = _run(code_dir, store=False)
    assert (after["key"], after["issues"]) == (plain["key"], plain["issues"])
    again = _run(code_dir, verify=True)
    assert again["stats"]["computed"] == 0
    assert (again["key"], again["issues"]) == (after["key"], after["issues"])


def test_a_global_bound_to_another_function_is_seen(code_dir):
    fresh = _run(code_dir)
    assert fresh["issues"] == []
    rebound = _run(code_dir, "rebind", verify=True)
    assert rebound["issues"] == ["scope_mutation: v[...] = ... - subscript mutation"]
    assert rebound["key"] != fresh["key"]
    plain = _run(code_dir, "rebind", store=False)
    assert (rebound["key"], rebound["issues"]) == (plain["key"], plain["issues"])
    back = _run(code_dir, verify=True)
    assert back["stats"]["computed"] == 0
    assert (back["key"], back["issues"]) == (fresh["key"], fresh["issues"])


def _corrupt_truncated(text: str) -> str:
    return text[: len(text) // 2]


def _corrupt_not_json(text: str) -> str:
    return "\x00garbage"


def _corrupt_wrong_shape(text: str) -> str:
    data = json.loads(text)
    data["texts"] = ["not", "a", "dict"]
    return json.dumps(data)


def _corrupt_bad_answers(text: str) -> str:
    """Indices past the end of the function, and names that are not names."""
    data = json.loads(text)
    for facts in data["texts"].values():
        for key in facts:
            facts[key] = [10_000] if "names" not in key and "locals" not in key else [1, 2]
    return json.dumps(data)


def _corrupt_other_path(text: str) -> str:
    data = json.loads(text)
    data["path"] = "/somewhere/else.py"
    return json.dumps(data)


@pytest.mark.parametrize(
    "corrupt",
    [_corrupt_truncated, _corrupt_not_json, _corrupt_wrong_shape, _corrupt_bad_answers, _corrupt_other_path],
    ids=["truncated", "not-json", "wrong-shape", "bad-answers", "other-path"],
)
def test_a_damaged_store_file_is_ignored(code_dir, corrupt):
    first = _run(code_dir)
    files = _store_files(code_dir)
    assert files
    for file in files:
        file.write_text(corrupt(file.read_text("utf-8")), encoding="utf-8")
    again = _run(code_dir, verify=True)
    assert again["stats"]["computed"] > 0
    assert (again["key"], again["issues"]) == (first["key"], first["issues"])
    # Written anew, whole: the next process loads it.
    third = _run(code_dir, verify=True)
    assert third["stats"]["computed"] == 0


def test_a_body_the_walk_rewrites_is_keyed_the_same_with_the_store(code_dir):
    """`spell_static_dispatch` spells ``globals()["make"](n)`` as
    ``make(n)`` in a new tree: the shared one, whose facts are stored, is
    still what the text parses to (the verify mode compares it)."""
    (code_dir / "mod.py").write_text(
        _MODULE.replace("    v = make(n)", '    v = globals()["make"](n)'), encoding="utf-8"
    )
    first = _run(code_dir, verify=True)
    assert first["key"] is not None
    second = _run(code_dir, verify=True)
    assert second["stats"]["loaded"] > 0
    assert (second["key"], second["issues"]) == (first["key"], first["issues"])
    plain = _run(code_dir, store=False, verify=True)
    assert (second["key"], second["issues"]) == (plain["key"], plain["issues"])


def test_spelling_a_lookup_leaves_the_shared_tree_as_parsed():
    import sys

    from cash.analysis import tree_memo
    from cash.analysis.static_dispatch import spell_static_dispatch

    src = 'def f(x):\n    y = [x]\n    return sys.modules["os"].path.join(globals()["g"](y))\n'
    tree = tree_memo.parse(src)
    before = ast.dump(tree, include_attributes=True)
    func_def = tree_memo.stripped_def(tree.body[0])
    spelled, issues = spell_static_dispatch(func_def, {"sys": sys, "g": len}, "f")
    assert issues == []
    assert spelled is not func_def
    assert "<module os>" in ast.dump(spelled) and "globals" not in ast.dump(spelled)
    assert ast.dump(tree, include_attributes=True) == before
    plain = tree_memo.stripped_def(tree.body[0])
    assert spell_static_dispatch(plain, {}, "f")[0] is not plain  # globals() is still the builtin
    same = tree_memo.parse("def h(x):\n    return x\n").body[0]
    assert spell_static_dispatch(same, {}, "h")[0] is same

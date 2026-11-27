"""What the purity rules read of a body's tree alone for its writer calls
(`purity_visitor._tree_writer_scope`: the names it binds to a writer, the
names its imports bind, what its loops iterate) and the imports written in
it (`helper_bindings._import_specs`) are facts of the tree, read back in a
fresh process for the same text (`tree_memo.fact`, `fact_store`).

A commit that adds a module constant leaves a cached function's stored
report stale, and the walk runs again over bodies that did not change: each
was walked whole once more for each of these. Now only a body whose text
changed works them out again; what the namespace binds (a global bound to
``setattr``, the package a relative import resolves against) is still read
live, and the key and the findings are the ones working it out gives.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from tests._scripts import run_python

_FACTS = ("writer_scope", "import_specs")

_RUNNER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    from cash.analysis import helper_walk, tree_memo
    counts = {name: {"asked": 0, "computed": 0} for name in %r}
    fact = tree_memo.fact
    run = helper_walk.HelperWalk.run
    walking = []

    def walk(self, *args, **kwargs):
        walking.append(True)
        try:
            return run(self, *args, **kwargs)
        finally:
            walking.pop()

    def counting(node, name, compute, stored=None):
        if name not in counts or not walking:
            return fact(node, name, compute, stored)
        counts[name]["asked"] += 1

        def counted(n):
            counts[name]["computed"] += 1
            return compute(n)

        return fact(node, name, counted, stored)

    tree_memo.fact = counting
    helper_walk.HelperWalk.run = walk
    import cash
    from cash.analysis.purity_analyzer import get_analyzer
    from pkg import mod
    key = cash.cache(mod.top).explain(2).cache_key
    issues = sorted(f"{i.kind}: {i.description}" for i in get_analyzer().analyze(mod.top).issues)
    print(json.dumps({"counts": counts, "key": key, "issues": issues}))
    """
    % (_FACTS,)
)

_MOD = """
class Box:
    pass


BOX = Box()


def helper(x):
    from json import dumps
    from .util import scale

    total = 0
    for row in [x]:
        total += row
    return len(dumps(total)) + scale(x)


def top(k: "int") -> int:
    return helper(k) + k
"""

_UTIL = """
def scale(x):
    return x * 2
"""

#: The body writes into a global through a name it binds to ``setattr``.
_WRITES = _MOD.replace("    total = 0\n", '    sa = setattr\n    sa(BOX, "k", x)\n    total = 0\n')
#: The body imports another function under the same name.
_REIMPORTS = _MOD.replace("from json import dumps", "from json import loads as dumps")


def _run(code_dir: Path, *, store: bool = True) -> dict:
    env = {"CASH_ANALYSIS_STORE": None if store else "0"}
    out = run_python("-c", _RUNNER, str(code_dir), cwd=code_dir, cache_dir=code_dir / ".cash", env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _commit(code_dir: Path, text: str, rev: int) -> None:
    """*text* with a module constant no function reads: the report goes
    stale and is walked again."""
    (code_dir / "pkg" / "mod.py").write_text(f"{text}\n_REV = {rev}\n", encoding="utf-8")


def _same_as_store_off(code_dir: Path, run: dict) -> None:
    off = _run(code_dir, store=False)
    assert (run["key"], run["issues"]) == (off["key"], off["issues"])


@pytest.fixture
def code_dir(tmp_path):
    code = tmp_path / "code"
    (code / "pkg").mkdir(parents=True)
    (code / ".cash").mkdir()
    (code / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (code / "pkg" / "util.py").write_text(_UTIL, encoding="utf-8")
    _commit(code, _MOD, 0)
    return code


def test_a_walk_after_an_unrelated_edit_reads_both_facts_back(code_dir):
    first = _run(code_dir)
    for name in _FACTS:
        assert first["counts"][name]["computed"] >= 1, first["counts"]
    _commit(code_dir, _MOD, 1)
    again = _run(code_dir)
    for name in _FACTS:
        assert again["counts"][name]["asked"] >= 1, f"control: the stale report was walked again: {again}"
        assert again["counts"][name]["computed"] == 0, again["counts"]
    assert (again["key"], again["issues"]) == (first["key"], first["issues"])
    _same_as_store_off(code_dir, again)


def test_a_writer_bound_in_the_edited_body_is_found(code_dir):
    first = _run(code_dir)
    _commit(code_dir, _WRITES, 1)
    edited = _run(code_dir)
    assert edited["counts"]["writer_scope"]["computed"] >= 1, edited["counts"]
    assert any("BOX" in issue for issue in edited["issues"]), edited["issues"]
    assert edited["issues"] != first["issues"]
    _same_as_store_off(code_dir, edited)


def test_an_import_changed_in_the_body_keys_anew(code_dir):
    first = _run(code_dir)
    _commit(code_dir, _REIMPORTS, 1)
    edited = _run(code_dir)
    assert edited["counts"]["import_specs"]["computed"] >= 1, edited["counts"]
    assert edited["key"] != first["key"]
    _same_as_store_off(code_dir, edited)


def test_an_edit_to_what_a_relative_import_in_the_body_names_keys_anew(code_dir):
    first = _run(code_dir)
    assert _run(code_dir)["key"] == first["key"]
    (code_dir / "pkg" / "util.py").write_text(_UTIL.replace("x * 2", "x * 3"), encoding="utf-8")
    edited = _run(code_dir)
    assert edited["counts"]["import_specs"]["computed"] == 0, "the body did not change"
    assert edited["key"] != first["key"]
    _same_as_store_off(code_dir, edited)

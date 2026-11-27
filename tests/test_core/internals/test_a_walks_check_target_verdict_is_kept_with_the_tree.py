"""The helper walk's verdict on whether a body names a class only to test
objects against it (`check_targets.named_only_as_check_target`) is a fact
of the body's tree, read back in a fresh process for the same text, class
name and builtin tests (`tree_memo.fact`, `fact_store`).

A commit that edits a file a cached function's report read outside the
helpers' own texts (a new module constant) leaves the report stale, and
the walk runs again over unchanged bodies: each class a body names was
tested again by parsing a parent map of the body. Now only a body whose
text changed, or whose tests are rebound, is worked out again, and the key
is the one working it out gives.
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
    from cash.analysis import tree_memo
    counts = {"asked": 0, "computed": 0}
    fact = tree_memo.fact

    def counting(node, name, compute, stored=None):
        if not (isinstance(name, tuple) and name[:1] == ("check_target_only",)):
            return fact(node, name, compute, stored)
        counts["asked"] += 1

        def counted(n):
            counts["computed"] += 1
            return compute(n)

        return fact(node, name, counted, stored)

    tree_memo.fact = counting
    import cash, mod
    key = cash.cache(mod.top).explain(2).cache_key
    print(json.dumps({"counts": counts, "key": key}))
    """
)

_MODULE = """
LIMIT = 3


class Kind:
    def size(self):
        return LIMIT


def helper(x):
    return 1 if isinstance(x, Kind) else 0


def top(k: "int") -> int:
    return helper(k) + k
"""

#: The helper reads the class as data: its verdict is False.
_AS_DATA = _MODULE.replace("return 1 if isinstance(x, Kind) else 0", "return Kind().size() + x")


def _run(code_dir: Path, *, store: bool = True) -> dict:
    env = {"CASH_ANALYSIS_STORE": None if store else "0"}
    out = run_python("-c", _RUNNER, str(code_dir), cwd=code_dir, cache_dir=code_dir / ".cash", env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _commit(code_dir: Path, text: str, rev: int) -> None:
    """*text* with a module constant no function reads: the report goes
    stale and is walked again; no helper's text changes."""
    (code_dir / "mod.py").write_text(f"{text}\n_REV = {rev}\n", encoding="utf-8")


@pytest.fixture
def code_dir(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / ".cash").mkdir()
    _commit(code, _MODULE, 0)
    return code


def test_a_walk_after_an_unrelated_edit_reads_the_verdict_back(code_dir):
    first = _run(code_dir)
    assert first["counts"]["computed"] >= 1, first["counts"]
    _commit(code_dir, _MODULE, 1)
    again = _run(code_dir)
    assert again["counts"]["asked"] >= 1, "control: the stale report was walked again"
    assert again["counts"]["computed"] == 0, again["counts"]
    assert again["key"] == first["key"] == _run(code_dir, store=False)["key"]


def test_an_edit_to_the_body_works_the_verdict_out_again(code_dir):
    first = _run(code_dir)
    _commit(code_dir, _AS_DATA, 1)
    edited = _run(code_dir)
    assert edited["counts"]["computed"] >= 1, edited["counts"]
    assert edited["key"] != first["key"]
    assert edited["key"] == _run(code_dir, store=False)["key"]
    # Read as data, LIMIT reaches the key.
    _commit(code_dir, _AS_DATA.replace("LIMIT = 3", "LIMIT = 4"), 2)
    assert _run(code_dir)["key"] != edited["key"]

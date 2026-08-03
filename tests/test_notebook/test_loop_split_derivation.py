"""The simulator and the runtime must derive the SAME split. Pinned here.

The failure this guards is silent. The re-execution planner runs the
statements the *simulator* modelled, so if the two sides ever derive
different halves, the planner re-runs one thing while entries exist for
another -- a stale value with nothing red anywhere. Three reverted attempts
at CAS-261 step 2 died on variations of exactly that.

These are unit tests of the derivation itself; the behavioural proof (an
upstream edit still invalidates a split loop) lives in the integration suite.
This file's job is to make the two sides' *inputs* provably identical.
"""
from __future__ import annotations

import ast
import json

import pytest

from cash.notebook.loop_split import (
    LoopSplitStore,
    get_store,
    is_split_half,
    loop_source_hash,
    split_nodes,
    split_sources,
)


def _for(src: str) -> ast.For:
    node = ast.parse(src).body[0]
    assert isinstance(node, ast.For)
    return node


SHAPES = [
    "for i in range(100):\n    result = result + 1",
    "for t in items:\n    out.append(compute(t))",
    "for a, b in pairs:\n    d[a] = b",
    "for x in list(range(1, 51)):\n    total += x",
    "for i in range(10):\n    for j in range(3):\n        acc.append(i * j)",
]


@pytest.mark.parametrize("src", SHAPES)
def test_split_sources_survive_a_source_round_trip(src):
    """The two sides never share an AST object: the runtime holds the node it
    is executing, the simulator re-parses the notebook's cell text. Identity
    therefore has to survive a trip through source, which is what re-parsing
    here simulates."""
    a = split_sources(_for(src), 5)
    assert split_sources(_for(src), 5) == a                    # independent parse
    assert split_sources(_for(ast.unparse(_for(src))), 5) == a  # unparse/reparse


def test_tail_slices_the_original_expression_not_a_temp_name():
    """A content digest of live data is uncomputable by the simulator, which
    has source and no values. Slicing the expression keeps the tail derivable
    from source alone -- the property the whole design rests on."""
    head, tail = split_sources(_for("for t in items:\n    out.append(f(t))"), 5)
    assert head == "for t in items[:5]:\n    out.append(f(t))"
    assert tail == "for t in items[5:]:\n    out.append(f(t))"


@pytest.mark.parametrize("k", [1, 5, 17])
def test_halves_cover_the_original_exactly_once_in_order(k):
    """Executed against a real list and compared with the undivided loop --
    the property that makes splitting semantics-preserving at all."""
    src = "for t in items:\n    out.append(t * 2)"
    whole = {"items": list(range(20)), "out": []}
    exec(compile(ast.Module(body=[_for(src)], type_ignores=[]), "<w>", "exec"), whole)

    head, tail = split_nodes(_for(src), k)
    split = {"items": list(range(20)), "out": []}
    exec(compile(ast.Module(body=[head, tail], type_ignores=[]), "<s>", "exec"), split)

    assert split["out"] == whole["out"]
    assert split["t"] == whole["t"], "leaked loop variable differs after a split"


def test_halves_are_recognisable_as_halves():
    """Both sides must refuse to split a half, or they recurse. Structural,
    not a marker attribute -- the simulator re-parses source and would lose
    any attribute the runtime set on its node."""
    head, tail = split_nodes(_for("for t in items:\n    f(t)"), 5)
    assert is_split_half(head) and is_split_half(tail)
    assert not is_split_half(_for("for t in items:\n    f(t)"))
    # A subscripted-but-not-sliced iterable is NOT a half.
    assert not is_split_half(_for("for t in groups[key]:\n    f(t)"))


def test_for_else_is_refused():
    """``else`` runs only if the loop completed without ``break``; a split
    loop has no single completion point. Refused loudly, not silently
    dropped."""
    with pytest.raises(ValueError):
        split_nodes(_for("for i in items:\n    pass\nelse:\n    done = True"), 5)


def test_source_hash_ignores_formatting_but_not_meaning():
    """Too strict and a reformatted cell loses its verdict (wasteful); too
    loose and an EDITED loop keeps a ``k`` chosen for different code, which
    is a correctness problem."""
    a = _for("for i in range(100):\n    result = result + 1")
    b = _for("for i in range(100):\n        result = result + 1")   # reindented
    c = _for("for i in range(100):\n    result = result + 2")       # edited
    assert loop_source_hash(a) == loop_source_hash(b)
    assert loop_source_hash(a) != loop_source_hash(c)


class TestLoopSplitStore:
    def test_roundtrips_a_verdict_across_instances(self, tmp_path):
        LoopSplitStore(str(tmp_path)).record("abc", 5)
        assert LoopSplitStore(str(tmp_path)).get("abc") == 5

    def test_never_rewrites_an_existing_verdict(self, tmp_path):
        """A ``k`` that moved between runs would change the tail's source and
        so its key -- the exact failure the store exists to prevent."""
        store = LoopSplitStore(str(tmp_path))
        store.record("abc", 5)
        store.record("abc", 9)
        assert store.get("abc") == 5
        assert LoopSplitStore(str(tmp_path)).get("abc") == 5

    def test_unknown_loop_has_no_verdict(self, tmp_path):
        assert LoopSplitStore(str(tmp_path)).get("nope") is None

    @pytest.mark.parametrize("payload", [
        "{ not json at all",
        json.dumps({"version": 999, "splits": {"abc": 5}}),
        json.dumps({"version": 1, "splits": "not a dict"}),
        json.dumps({"version": 1, "splits": {"abc": -1}}),
        json.dumps({"version": 1, "splits": {"abc": "five"}}),
    ])
    def test_a_broken_store_means_no_split_not_a_crash(self, tmp_path, payload):
        """Failure mode must be "no optimisation", never a raise or a wrong
        answer."""
        (tmp_path / "_loop_split.json").write_text(payload, encoding="utf-8")
        assert LoopSplitStore(str(tmp_path)).get("abc") is None

    def test_no_cache_dir_degrades_to_session_scoped(self):
        store = LoopSplitStore(None)
        store.record("abc", 5)
        assert store.get("abc") == 5

    def test_get_store_returns_one_shared_instance(self, tmp_path):
        """Load-bearing, not an optimisation: two instances diverge the moment
        a verdict is recorded, because each loads from disk only once. The
        runtime would then record a split the simulator never applies."""
        from cash.notebook.loop_split import _reset_stores_for_tests
        _reset_stores_for_tests()
        a = get_store(str(tmp_path))
        b = get_store(str(tmp_path))
        assert a is b
        a.record("abc", 5)
        assert b.get("abc") == 5, "a second lookup did not see a just-recorded verdict"
        _reset_stores_for_tests()

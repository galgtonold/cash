"""A library call that changes what it is handed, straight inside an ``if``,
a ``try`` or a ``with`` block, moves what it changed.

A ``with`` block runs as one unit and is watched as a loop is. An ``if`` or a
``try`` runs statement by statement: it is moved by what its statements were
seen changing, and, for one that has not run yet (a branch not taken), by
every name it hands to such code.
"""

from __future__ import annotations

import ast
import collections
import heapq

import numpy as np

from cash.analysis.mutation_effects import loop_watch_candidates

NAMESPACE = {"heapq": heapq, "np": np, "h": [], "a": np.zeros(2), "dq": collections.deque([1, 2]), "K": 1, "f": None}


def _is_module(name: str) -> bool:
    return name in {"heapq", "np"}


def _node(code: str) -> ast.AST:
    return ast.parse(code).body[0]


def test_a_with_block_is_watched_for_what_it_hands_a_library():
    assert loop_watch_candidates(
        _node("with np.errstate(all='ignore'):\n    np.copyto(a, K)"), NAMESPACE, _is_module
    ) == {"a"}


def test_what_a_with_block_binds_is_not_watched():
    code = "with open('x') as f:\n    data = f.read()\n    heapq.heappush(h, 1)"
    assert loop_watch_candidates(_node(code), NAMESPACE, _is_module) == {"h"}


def test_a_branch_is_moved_by_what_its_statements_were_seen_changing():
    from cash.notebook.cache_key import statement_source_hash
    from cash.notebook.loop_watch import branch_changes

    node = _node("if K > 1:\n    job = dq.popleft()\nelse:\n    heapq.heappush(h, K)")
    # Not run yet: every name either branch hands to such code.
    assert branch_changes(node, lambda key: None, NAMESPACE, _is_module, None) == {"dq", "h"}
    # The else branch ran and was seen changing h; the if branch never ran.
    recorded = {statement_source_hash("heapq.heappush(h, K)"): {"h"}}
    assert branch_changes(node, recorded.get, NAMESPACE, _is_module, None) == {"dq", "h"}
    recorded[statement_source_hash("job = dq.popleft()")] = set()
    assert branch_changes(node, recorded.get, NAMESPACE, _is_module, None) == {"h"}


def test_a_try_reads_the_verdict_of_a_with_block_inside_it():
    from cash.notebook.loop_watch import branch_changes, loop_watch_hash

    inner = "with np.errstate(all='ignore'):\n    np.copyto(a, K)"
    node = _node("try:\n" + "\n".join("    " + line for line in inner.splitlines()) + "\nexcept ValueError:\n    pass")
    recorded = {loop_watch_hash(ast.unparse(_node(inner))): {"a"}}
    assert branch_changes(node, recorded.get, NAMESPACE, _is_module, None) == {"a"}


def test_a_loop_or_with_block_is_not_moved_through_its_statements():
    from cash.notebook.loop_watch import branch_changes

    assert (
        branch_changes(
            _node("for i in range(2):\n    heapq.heappush(h, i)"), lambda key: None, NAMESPACE, _is_module, None
        )
        == set()
    )

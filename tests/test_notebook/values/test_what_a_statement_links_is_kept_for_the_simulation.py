"""What a statement links is logged, replayed, and read from its text when
no run says.

The edges between names that share an object lived only in the kernel that
ran the cells, and the simulation read the kernel's edges as they ended: a
restart left none (``y.append(2)`` no longer changed ``x``), an edit that made
an alias (``d2 = df``) added none, and one that removed it kept the old one.
"""

from __future__ import annotations

import ast

# The module, not its names: each test fails on its own without them.
from cash.notebook.statement import derivation_edges as de

SOURCES = {
    "ok": "def ok(v):\n    return v\n",
    "first": "def first(rows, n=1):\n    return rows[0]\n",
    "double": "def double(v):\n    return v * 2\n",
}


def _static(code: str, namespace=None) -> set[tuple[str, str]]:
    return set(de.static_share_edges(ast.parse(code), SOURCES.get, namespace or {}))


def both(a: str, b: str) -> set[tuple[str, str]]:
    return {(a, b), (b, a)}


def test_the_text_links_a_name_to_what_its_value_may_be():
    assert _static("y = x") == both("y", "x")
    assert _static("y = x[0]") == both("y", "x")
    assert _static("head = big[:3]") == both("head", "big")
    assert _static("runs = [cfg]") == both("runs", "cfg")
    assert _static("a, b = x, w") == both("a", "x") | both("b", "w")
    assert _static("y = ok(x)") == both("y", "x")
    assert _static("y = first(rows)") == both("y", "rows")


def test_a_new_object_links_nothing():
    """Control: copies, other calls, and numbers."""
    assert _static("y = list(x)") == set()
    assert _static("d2 = df.copy()") == set()
    assert _static("y = double(x)") == set()
    assert _static("m = n", {"n": 5}) == set()


def test_a_log_takes_every_edge_linked_inside_it_nested_too():
    edges: dict[str, set[str]] = {}
    with de.logging_edges() as outer:
        de.link(edges, "a", "b")
        with de.logging_edges() as inner:
            de.record_shared_object_edges(edges, ["y"], ["x"])
    de.link(edges, "c", "d")
    assert inner == both("y", "x")
    assert outer == {("a", "b")} | both("y", "x")
    assert edges["c"] == {"d"}


def test_two_equal_logs_close_in_their_own_order():
    with de.logging_edges() as first:
        with de.logging_edges() as second:
            pass
        de.link({}, "a", "b")
    assert first == {("a", "b")} and second == set()


def test_a_replay_drops_the_rebound_names_edges_before_adding_its_own():
    edges = {"y": {"x"}, "x": {"y"}}
    de.apply_share_record(edges, ["y"], both("y", "w"))
    assert edges == {"y": {"w"}, "w": {"y"}}

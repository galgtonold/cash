"""A short loop whose body holds a loop runs as one unit when the two together
are long.

A loop of 40 over log lines, each with an inner loop over the line's actions,
ran 1,100 body statements one at a time: 18 s of per-statement cost around
5 ms of work. The policy sized the loop by its 40 iterations alone, below the
50 it needs. The inner loop is typically sized by a variable the outer
iteration binds, so it is assumed to run ten times per pass.
"""

from __future__ import annotations

import ast

from cash.notebook.control_structures import single_unit_policy as policy


def _single_unit(source: str, user_ns: dict) -> bool:
    node = ast.parse(source).body[0]
    iterable = eval(ast.unparse(node.iter), dict(user_ns), dict(user_ns))
    return policy.should_run_as_single_unit(node, iterable, user_ns)


FLAT = "for line in data[:40]:\n    parts = line.split()\n    name = parts[0]\n    rows.append(name)\n"
NESTED = (
    "for line in data[:40]:\n"
    "    parts = line.split()\n"
    "    for action in parts[1:]:\n"
    "        name = action.strip()\n"
    "        rows.append(name)\n"
)
ENUMERATED = NESTED.replace("for line in data[:40]:", "for i, line in enumerate(data[:40]):")


def test_forty_iterations_with_a_loop_inside_run_as_one_unit():
    user_ns = {"data": ["a b c d"] * 100, "rows": []}

    assert _single_unit(NESTED, user_ns)


def test_forty_iterations_of_a_flat_body_keep_their_per_iteration_entries():
    user_ns = {"data": ["a b c d"] * 100, "rows": []}

    assert not _single_unit(FLAT, user_ns)


def test_an_enumerated_slice_is_sized_through_the_slice():
    user_ns = {"data": ["a b c d"] * 100, "rows": []}
    node = ast.parse(ENUMERATED).body[0]
    iterable = eval(ast.unparse(node.iter), dict(user_ns), dict(user_ns))

    assert policy.estimated_iterations(node.iter, iterable, user_ns) == 40
    assert policy.should_run_as_single_unit(node, iterable, user_ns)


def test_a_loop_of_ten_with_a_loop_inside_stays_per_iteration():
    user_ns = {"data": ["a b c d"] * 100, "rows": []}

    assert not _single_unit(NESTED.replace("[:40]", "[:4]"), user_ns)

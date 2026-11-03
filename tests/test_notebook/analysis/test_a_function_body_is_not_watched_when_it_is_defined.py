"""A ``def`` statement hands nothing in its body to anyone: the body runs
when the function is called, on its own parameters.

``@cash.cache def total(points): return sum(p.scaled() for p in points)``
was read as handing the notebook's ``points`` to ``scaled`` item by item, so
the ``def`` was watched as one that may change ``points`` in place. After
the class cell above it was edited, the upstream check counted the ``def``
as a writer of ``points`` and refused the plan as one that would leave the
cell an earlier ``points`` (``NOTEBOOK-REPAIR-GAP``).
"""

from __future__ import annotations

import ast

import pytest

from cash.analysis.mutation_effects import comprehension_watch_candidates, is_module_name


class _Point:
    def scaled(self):
        return 1


def _candidates(code: str) -> frozenset[str]:
    ns = {"points": [_Point()], "rows": [[1]]}
    return comprehension_watch_candidates(
        ast.parse(code), ns, lambda name: is_module_name(name, ns, ()), lambda name: None
    )


@pytest.mark.parametrize(
    "code",
    [
        "def total(points):\n    return sum(p.scaled() for p in points)",
        "@cash.cache\ndef total(points):\n    return sum(p.scaled() for p in points)",
        "async def total(points):\n    return sum(p.scaled() for p in points)",
        "class Model:\n    def total(self):\n        return [p.scaled() for p in points]",
    ],
    ids=["def", "decorated", "async", "method"],
)
def test_a_function_body_hands_nothing_when_defined(code):
    assert _candidates(code) == frozenset()


@pytest.mark.parametrize(
    "code",
    [
        "s = sum(p.scaled() for p in points)",
        "def total(xs=[p.scaled() for p in points]):\n    return xs",
        "class Model:\n    scaled = [p.scaled() for p in points]",
    ],
    ids=["statement", "default", "class-body"],
)
def test_what_runs_when_the_statement_runs_is_still_watched(code):
    """Control: a comprehension that runs now, at the top level, in a
    default or in a class body, still hands ``points`` over."""
    assert _candidates(code) == frozenset({"points"})

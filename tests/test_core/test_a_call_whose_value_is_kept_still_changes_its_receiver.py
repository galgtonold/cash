"""A call that changes its receiver changes it whether or not its value is kept.

``label = df.pop('label')``, ``print(rows.pop())`` and ``del df['label']``
change the name in place, as ``df.pop('label')`` on its own line does: the
name's lineage moves with them, or a statement reading it next is served the
entry it had before the change.
"""

from __future__ import annotations

import ast

import pytest

from cash.analysis.mutation_effects import classify_receivers


def _mutated(code: str, namespace: dict | None = None) -> frozenset[str]:
    ns = {"df": {}, "rows": [], "d": {}} if namespace is None else namespace
    return classify_receivers(ast.parse(code), ns, lambda: None, arguments=()).mutated


@pytest.mark.parametrize(
    ("code", "name"),
    [
        ("label = df.pop('label')", "df"),
        ("print(rows.pop())", "rows"),
        ("del df['label']", "df"),
        ("x: list = d.setdefault('k', [])", "d"),
        ("n = len([rows.append(i) for i in range(3)])", "rows"),
        ("out = df.drop(columns=['a'], inplace=True)", "df"),
    ],
)
def test_the_receiver_is_changed(code, name):
    assert name in _mutated(code)


@pytest.mark.parametrize(
    "code",
    [
        "m = df.get('label')",
        "s = rows.sort('ts').head()",
        "y = df.drop(columns=['a'])",
        "del df",
    ],
)
def test_a_call_that_leaves_it_alone_changes_nothing(code):
    assert not _mutated(code)


def test_a_module_is_not_a_receiver():
    import os

    assert "os" not in _mutated("v = os.environ.pop('X', None)", {"os": os})

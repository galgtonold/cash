"""Which single-unit loops may store the containers they fill in place.

`seeded_accumulator_loop` lifts the in-place refusal only for containers
seeded empty earlier in the same cell and used inside the loop only to grow,
read or change what they hold. Everything else keeps the refusal.
"""

import ast

import pytest

from cash.analysis.mutations import seeded_accumulator_loop


def _detect(src: str):
    body = ast.parse(src).body
    i = max(n for n, node in enumerate(body) if isinstance(node, ast.For))
    return seeded_accumulator_loop(body[i], body[:i])


@pytest.mark.parametrize(
    "src, expected",
    [
        pytest.param("out = []\nfor t in xs:\n    out.append(f(t))", {"out", "t"}, id="one-append"),
        pytest.param(
            "df2 = df.sort_values('t')\ndeltas = []\nfor _, g in df2.groupby('u'):\n    for d in g['t']:\n        deltas.append(d)",
            {"deltas", "_", "g", "d"},
            id="nested-loop-statement-between",
        ),
        pytest.param(
            "a, b = [], set()\nn = 0\nfor x in xs:\n    if not x:\n        continue\n    n += 1\n    a.append(x)\n    b.add(n)",
            {"a", "b", "x"},
            id="tuple-seed-continue-counter",
        ),
        pytest.param("d = {}\nfor k in xs:\n    d[k] = d.get(k, 0) + 1", {"d", "k"}, id="dict-counter"),
        pytest.param("g = {}\nfor k, v in xs:\n    g.setdefault(k, []).append(v)", {"g", "k", "v"}, id="setdefault"),
        pytest.param(
            "out = []\nfor t in xs:\n    out += [t]\n    if len(out) > 3 and t in out:\n        break",
            {"out", "t"},
            id="aug-and-reads",
        ),
    ],
)
def test_fresh_containers_are_kept(src, expected):
    assert _detect(src) == frozenset(expected)


@pytest.mark.parametrize(
    "src",
    [
        pytest.param("for t in xs:\n    out.append(t)", id="seeded-in-an-earlier-cell"),
        pytest.param("out = [1]\nfor t in xs:\n    out.append(t)", id="seeded-non-empty"),
        pytest.param("out = []\nout.append(0)\nfor t in xs:\n    out.append(t)", id="filled-after-seed"),
        pytest.param("out = []\nb = out\nfor t in xs:\n    out.append(t)", id="aliased-after-seed"),
        pytest.param("a = b = []\nfor t in xs:\n    a.append(t)", id="two-names-one-object"),
        pytest.param("out = []\nfor t in xs:\n    f(out)", id="handed-to-a-callee"),
        pytest.param("out = []\nfor t in xs:\n    other.append(out)", id="stored-elsewhere"),
        pytest.param("out = []\nfor t in xs:\n    ap = out.append\n    ap(t)", id="bound-method"),
        pytest.param("out = []\nfor t in out:\n    out.append(t)", id="iterates-itself"),
        pytest.param("out = []\nfor t in xs:\n    out = out + [t]", id="rebound"),
        pytest.param("out = []\nfor t in xs:\n    out -= [t]", id="wrong-operator"),
        pytest.param("out = []\nfor t in xs:\n    out.append(t)\n    temp[t] = 1", id="mutates-a-foreign-object"),
        pytest.param("for k in temp.columns:\n    temp[k] = temp[k] * 2", id="column-assign"),
        pytest.param("out = []\nfor t in xs:\n    def g():\n        pass\n    out.append(t)", id="defines-a-function"),
        pytest.param("out = []\nfor a, *b in xs:\n    out.append(a)", id="starred-target"),
        pytest.param("out = []\nfor t in xs:\n    for o.x in t:\n        out.append(o.x)", id="attribute-target"),
        pytest.param("n = 0\nfor t in xs:\n    n += t", id="no-container"),
    ],
)
def test_everything_else_keeps_the_refusal(src):
    assert _detect(src) is None

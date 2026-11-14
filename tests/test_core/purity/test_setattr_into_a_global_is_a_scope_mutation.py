"""``setattr`` and its kind writing into a global is a scope mutation, however the call is used.

``[setattr(b, "x", v) for b in BOXES]``, ``r = setattr(REG.e, "x", v)``,
``return setattr(...) or v * 2`` and ``object.__setattr__(REG.e, "x", v)``
were cached with no warning at all, and a bare ``setattr(REG.e, "x", v)``
(in the cached function or in a helper in another module) was warned only
because its return was dropped. A hit skips the write each time, so each
must say so on the first call, as ``REG.e.x = v`` does.
"""

from __future__ import annotations

import operator
import re
import textwrap
from operator import setitem as put

import pytest

from cash.analysis.purity_analyzer import PurityAnalyzer
from cash.analysis.purity_report import ISSUE_SCOPE_MUTATION
from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]


class _Box:
    pass


class _Registry:
    pass


BOXES = [_Box(), _Box()]
REG = _Registry()
REG.e = _Box()
TABLE: dict = {}
osa = object.__setattr__


def in_a_comprehension(v):
    [setattr(b, "x", v) for b in BOXES]
    return v


def in_a_loop(v):
    for b in BOXES:
        setattr(b, "x", v)
    return v


def its_value_bound(v):
    r = setattr(REG.e, "x", v)
    return v if r is None else r


def in_a_return(v):
    return setattr(REG.e, "x", v) or v * 2


def through_object(v):
    r = object.__setattr__(REG.e, "x", v)
    return v if r is None else r


def as_a_statement(v):
    setattr(REG.e, "x", v)
    return v


def deleting(v):
    delattr(REG.e, "x")
    return v


def through_operator(v):
    operator.setitem(TABLE, "k", v)
    return v


def through_an_imported_name(v):
    put(TABLE, "k", v)
    return v


def through_a_bound_name(v):
    osa(REG.e, "x", v)
    return v


def through_map(v):
    list(map(setattr, BOXES, ["x", "x"], [v, v]))
    return v


def into_a_captured_variable():
    box = _Box()

    def write(v):
        setattr(box, "x", v)
        return v

    return write


@pytest.mark.parametrize(
    "func",
    [
        in_a_comprehension,
        in_a_loop,
        its_value_bound,
        in_a_return,
        through_object,
        as_a_statement,
        deleting,
        through_operator,
        through_an_imported_name,
        through_a_bound_name,
        through_map,
        into_a_captured_variable(),
    ],
)
def test_a_write_into_a_global_or_a_captured_variable_is_a_scope_mutation(func):
    found = [i for i in PurityAnalyzer().analyze(func).issues if i.kind == ISSUE_SCOPE_MUTATION]
    assert found, func.__name__


def into_a_local(v):
    box = _Box()
    setattr(box, "x", v)
    r = object.__setattr__(box, "y", v)
    return box.x, r


def into_a_local_table(v):
    table = {}
    operator.setitem(table, "k", v)
    return table


def into_the_argument(box, v):
    setattr(box, "x", v)
    return v


def into_an_element_of_the_argument(boxes, v):
    [setattr(b, "x", v) for b in boxes]
    return v


@pytest.mark.parametrize("func", [into_a_local, into_a_local_table, into_the_argument, into_an_element_of_the_argument])
def test_a_write_into_a_local_or_an_argument_is_not_a_global_write(func):
    found = [
        i
        for i in PurityAnalyzer().analyze(func).issues
        if i.kind == ISSUE_SCOPE_MUTATION and "outside the function" in i.description
    ]
    assert not found, [i.description for i in found]


HELPER = textwrap.dedent("""
    class Entry:
        pass

    class Registry:
        pass

    REG = Registry()
    REG.e = Entry()

    def write_it(v):
        setattr(REG.e, "x", v)
""")

MAIN = textwrap.dedent("""
    import warnings
    import cash
    import helper

    class Box:
        pass

    BOXES = [Box(), Box()]
    RUNS = []

    @cash.cache
    def stamp(v):
        RUNS.append(v)  # @cash:assume-safe
        [setattr(b, "x", v) for b in BOXES]
        return v * 2

    @cash.cache
    def through_helper(v):
        helper.write_it(v)
        return v

    @cash.cache(strict=True)
    def strict(v):
        setattr(helper.REG.e, "y", v)
        return v

    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter("always")
        assert stamp(1) == 2
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        assert stamp(1) == 2
        assert stamp(1) == 2
    print("FIRST", [str(w.message).split(":")[0] for w in first])
    print("AGAIN", len(again))
    print("RUNS", len(RUNS))

    with warnings.catch_warnings(record=True) as helped:
        warnings.simplefilter("always")
        through_helper(1)
    print("HELPER", len(helped), [w["message"] for w in through_helper.cache_info()["warnings"]])

    try:
        strict(1)
    except cash.CashImpureFunctionError as exc:
        print("STRICT RAISED", "setattr" in str(exc))
    else:
        print("STRICT RAN")
""")


def test_the_decorator_warns_on_the_first_call_and_raises_under_strict(tmp_path):
    (tmp_path / "helper.py").write_text(HELPER, encoding="utf-8")
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    proc = run_python("main.py", cwd=tmp_path, env={"PYTHONWARNINGS": "always"})
    out = proc.stdout
    assert proc.returncode == 0, proc.stderr
    assert re.search(r"FIRST \[.*IMPURE-SIDE-EFFECTS", out), out
    # Warned once; the result is stored and served, the write skipped.
    assert "AGAIN 0" in out, out
    runs = int(re.search(r"RUNS (\d+)", out).group(1))
    assert runs <= 2, out
    assert re.search(r"HELPER 1 \[.*helper\.write_it.*setattr\(REG\.e, \.\.\.\)", out, re.S), out
    assert "STRICT RAISED True" in out, out

"""A statement reaching a chain of the user's functions walks it once, not once per function.

A user module where ``f0`` calls ``f1`` ... ``fN``: Run All took 1.9 s at
N=50, 5.4 s at 100 and 21.7 s at 200, as each function the statement
reaches was walked as a root of its own, every helper below it again, for
the environment and the clock it reads; and what each hands to code whose
effect is unknown was followed down the chain once per caller. A
function a caller's walk audited whole now adds nothing walked again, and
neither does a hand-off already followed, with the same answers.
"""

import os
import sys

import pytest

from cash.analysis import helper_walk, namespace_effects
from cash.analysis.purity_analyzer import PurityAnalyzer
from cash.notebook import call_key
from cash.notebook.callee_reach import reached_user_code
from cash.notebook.lineage_formula import callee_environment_reads, helper_ambient_reads

MAIN = """\
import os, time, uuid
import cash
import {other}

def f0(x):
    return f1(x) + len(os.environ.get("CASH_CC_A", ""))

def f1(x):
    return f2(x) + time.time()

def f2(x):
    return x

@cash.pure
def p(x):
    return q(x)

def q(x):
    return os.environ.get("CASH_CC_B") or uuid.uuid4().hex

def uses_p(x):
    return p(x), q(x)

def in_a_class_body():
    return os.environ.get("CASH_CC_F")

class K:
    X = in_a_class_body()

def both():
    return K.X, in_a_class_body()

def both_after():
    return in_a_class_body(), K.X

def stamp():
    return time.time()

def judged(x):
    t0 = stamp()
    fn = stamp
    return fn() - t0

def rec(x):
    return rec(x - 1) if x else os.getcwd()

def make():
    y = 3
    def inner():
        return os.environ.get("CASH_CC_D", y)
    return inner

closure = make()

def calls_closure():
    return closure()

def cross(x):
    return {other}.g(x) + f0(x)
"""

OTHER = """\
import os, time

def g(x):
    return h(x)

def h(x):
    return len(os.environ.get("CASH_CC_E", "")) + time.time()
"""

STATEMENTS = [
    "y = m.f0(1)",
    "y = m.f2(1) + m.f1(1) + m.f0(1)",
    "y = m.uses_p(1)",
    "y = m.p(1) + m.uses_p(1)",
    "y = m.both()",
    "y = m.both_after()",
    "y = m.judged(1)",
    "y = m.stamp() + m.judged(1)",
    "y = m.rec(3)",
    "y = m.calls_closure()",
    "y = m.cross(1)",
]


@pytest.fixture
def module(tmp_path, monkeypatch):
    tag = f"{os.getpid()}_{id(tmp_path)}"
    main, other = f"_ccmain_{tag}", f"_ccother_{tag}"
    (tmp_path / f"{other}.py").write_text(OTHER, encoding="utf-8")
    (tmp_path / f"{main}.py").write_text(MAIN.format(other=other), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield __import__(main)
    sys.modules.pop(main, None)
    sys.modules.pop(other, None)


def _one_walk_per_function(code, user_ns):
    """What the readers gave when they analysed every function on its own."""
    from cash.analysis.purity_report import ISSUE_AMBIENT_READ
    from cash.notebook.lineage_formula import _DURATION_CLOCKS, _makes_the_read

    reads, ambient = set(), {}
    for fn in reached_user_code(code, user_ns).functions:
        report = PurityAnalyzer().analyze(fn)
        reads |= report.environment_reads
        for issue in report.issues:
            mark = (issue.where, issue.description)
            if issue.kind != ISSUE_AMBIENT_READ or issue.description.startswith(_DURATION_CLOCKS):
                continue
            if mark not in ambient or _makes_the_read(fn, issue.where):
                ambient[mark] = (fn, issue)
    return reads, list(ambient.values())


@pytest.mark.parametrize("code", STATEMENTS)
def test_the_reads_found_are_those_of_a_walk_per_function(module, code):
    user_ns = {"m": module}
    reads, ambient = _one_walk_per_function(code, user_ns)
    assert callee_environment_reads(code, user_ns) == reads
    assert helper_ambient_reads(code, user_ns) == ambient


def test_every_shape_reads_something(module):
    """The comparison above is not empty against empty."""
    for code in STATEMENTS:
        reads, ambient = _one_walk_per_function(code, {"m": module})
        assert reads or ambient, code


CHAIN = 60


@pytest.fixture
def chain(tmp_path, monkeypatch):
    name = f"_ccchain_{os.getpid()}_{id(tmp_path)}"
    src = "import os\n" + "".join(f"def f{i}(x):\n    return f{i + 1}(x) + 1\n" for i in range(CHAIN))
    src += f"def f{CHAIN}(x):\n    return len(os.environ.get('CASH_CC_CHAIN', '')) + x\n"
    (tmp_path / f"{name}.py").write_text(src, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield __import__(name)
    sys.modules.pop(name, None)


def _counting(monkeypatch, owner, attr):
    calls = []
    real = getattr(owner, attr)

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, attr, counted)
    return calls


def test_a_chain_is_walked_once_for_the_environment_it_reads(chain, monkeypatch):
    walked = _counting(monkeypatch, helper_walk.HelperWalk, "_walk_one")
    reads = callee_environment_reads("y = m.f0(1)", {"m": chain})
    assert reads == {("env", "CASH_CC_CHAIN")}
    assert len(walked) == CHAIN + 1


def test_a_chain_is_followed_once_for_what_it_hands_on(chain, monkeypatch):
    monkeypatch.setattr(namespace_effects, "_HANDOFFS", {})
    followed = _counting(monkeypatch, namespace_effects, "_handoffs")
    watched = call_key.callee_watched_globals(chain.f0, ())
    assert watched == ()
    assert len(followed) <= 2 * (CHAIN + 1)


def test_a_recursive_helper_is_still_cut_where_it_calls_itself(tmp_path, monkeypatch):
    """The cut answer depends on what was seen: kept apart from a whole one."""
    import ast

    source = "def push(h, x):\n    if x:\n        push(h, x - 1)\n    h.append(x)\n"
    fdef = ast.parse(source).body[0]
    namespace = {}
    sources = {"push": source}.get
    first = namespace_effects.function_handoffs(fdef, namespace, sources)
    again = namespace_effects.function_handoffs(fdef, namespace, sources, frozenset({"push"}))
    assert first == again

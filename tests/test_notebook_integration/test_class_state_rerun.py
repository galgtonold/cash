"""Hidden mutation of shared class state via a constructor or instance method must
reset on isolated re-run (CAS-73, extends CAS-68). Constructing an object or
calling an instance method mutates a class variable (``Reg.registry``,
``Shared.data``) or an instance attribute reached through a method whose body
cash does not analyse; on an isolated re-run the state accumulates.

Each cell mutates ONCE, so the correct value is identical on the first run and
every re-run.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _rerun(nb_runner, setup, cell, expect):
    nb_runner.create_notebook([setup, cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expect in nb_runner.get_output(2), f"first: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert expect in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"


# --- must reset ---------------------------------------------------------------

def test_init_appends_class_var(nb_runner):
    _rerun(nb_runner,
           "class Reg:\n    registry = []\n    def __init__(self):\n        Reg.registry.append(id(self))",
           "r = Reg()\nprint('len', len(Reg.registry))", "len 1")


def test_shared_class_attr_sibling_observed(nb_runner):
    _rerun(nb_runner,
           "class Shared:\n    data = []\n    def add(self, x):\n        self.data.append(x)\na = Shared()\nb = Shared()",
           "a.add('v')\nprint('len', len(b.data))", "len 1")


def test_instance_method_appends_and_returns(nb_runner):
    _rerun(nb_runner,
           "class Stack:\n    def __init__(self):\n        self.data = []\n    def push(self, x):\n        self.data.append(x)\n        return self.data\ns = Stack()",
           "r = s.push('x')\nprint('len', len(s.data))", "len 1")


@pytest.mark.xfail(reason="CAS-75: cross-cell class-var accumulator needs a "
                          "per-cell snapshot; class-def re-run is suppressed to "
                          "avoid clobbering the upstream cell's contribution",
                   strict=False)
def test_init_increments_class_counter(nb_runner):
    # A class counter incremented by an UPSTREAM cell (w0) AND this cell (w): the
    # class-def reset is suppressed (it would clobber w0), so the re-run still
    # accumulates. See CAS-75.
    _rerun(nb_runner,
           "class Widget:\n    count = 0\n    def __init__(self):\n        Widget.count += 1\nw0 = Widget()",
           "w = Widget()\nprint('count', Widget.count)", "count 2")


# --- must NOT over-invalidate (pure) ------------------------------------------

def test_pure_construction_not_over_invalidated(nb_runner):
    _rerun(nb_runner,
           "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n    def norm(self):\n        return self.x + self.y",
           "p = Point(3, 4)\nprint('n', p.norm())", "n 7")


def test_pure_instance_method_not_over_invalidated(nb_runner):
    # instance method reads instance state but mutates nothing persistent.
    _rerun(nb_runner,
           "class Acc:\n    def __init__(self):\n        self.total = 10\n    def scaled(self, k):\n        return self.total * k\nacc = Acc()",
           "r = acc.scaled(3)\nprint('r', r)", "r 30")

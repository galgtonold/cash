"""Hidden mutation through the descriptor protocol must reset on isolated re-run
(CAS-77, extends CAS-70). An attribute ASSIGN (``c.x = v``) or attribute LOAD
(``v = m.now``) dispatches to a user ``@property`` setter/getter or a
data-descriptor ``__set__`` / ``__get__`` whose body mutates hidden state; on an
isolated re-run it accumulates.

Each access runs the accessor ONCE, so the correct value is identical on the
first run and every re-run.
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

def test_property_setter_free_var(nb_runner):
    _rerun(nb_runner,
           "log = []\nclass Cfg:\n    @property\n    def x(self):\n        return self._x\n    @x.setter\n    def x(self, v):\n        log.append(v)\n        self._x = v\nc = Cfg()",
           "c.x = 5\nprint('log', len(log))", "log 1")


def test_property_setter_self_list(nb_runner):
    _rerun(nb_runner,
           "class Rec:\n    def __init__(self):\n        self.history = []\n    @property\n    def val(self):\n        return self._v\n    @val.setter\n    def val(self, v):\n        self.history.append(v)\n        self._v = v\nr = Rec()",
           "r.val = 7\nprint('hist', len(r.history))", "hist 1")


def test_property_getter_side_effect_self(nb_runner):
    _rerun(nb_runner,
           "class Meter:\n    def __init__(self):\n        self.reads = []\n    @property\n    def now(self):\n        self.reads.append(1)\n        return len(self.reads)\nm = Meter()",
           "v = m.now\nprint('reads', len(m.reads))", "reads 1")


def test_descriptor_set_free_var(nb_runner):
    _rerun(nb_runner,
           "log = []\nclass Tracked:\n    def __set__(self, obj, v):\n        log.append(v)\n        obj.__dict__['_v'] = v\n    def __get__(self, obj, owner=None):\n        return obj.__dict__.get('_v') if obj is not None else self\nclass Model:\n    field = Tracked()\nmdl = Model()",
           "mdl.field = 3\nprint('log', len(log))", "log 1")


def test_descriptor_get_side_effect_free_var(nb_runner):
    _rerun(nb_runner,
           "log = []\nclass Logged:\n    def __get__(self, obj, owner=None):\n        log.append(1)\n        return len(log)\nclass Model2:\n    field = Logged()\nmdl2 = Model2()",
           "v = mdl2.field\nprint('log', len(log))", "log 1")


# --- must NOT over-invalidate (pure) ------------------------------------------

def test_pure_property_not_over_invalidated(nb_runner):
    _rerun(nb_runner,
           "class Circle:\n    def __init__(self, r):\n        self._r = r\n    @property\n    def area(self):\n        return 3 * self._r * self._r\nc = Circle(2)",
           "a = c.area\nprint('a', a)", "a 12")


def test_plain_attribute_assign_not_over_invalidated(nb_runner):
    # a plain attribute assign (no property/descriptor) must not be flagged
    _rerun(nb_runner,
           "class Bag:\n    def __init__(self):\n        self.n = 0\nbag = Bag()",
           "bag.n = 5\nprint('n', bag.n)", "n 5")

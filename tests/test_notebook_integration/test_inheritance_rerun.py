"""Hidden mutation reached through INHERITANCE must reset on isolated re-run
(CAS-76, extends the object-protocol engine / CAS-73). A mutation performed by an
inherited ``__init__`` / method / ``__enter__``, or on a class variable owned by a
base class, is invisible to the receiver's own class body; on an isolated re-run
it accumulates.

Each helper mutates ONCE, so the correct value is identical on the first run and
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

def test_inherited_init_appends_class_var(nb_runner):
    _rerun(nb_runner,
           "class Base:\n    registry = []\n    def __init__(self):\n        Base.registry.append(1)\nclass Sub(Base):\n    pass",
           "s = Sub()\nprint('len', len(Base.registry))", "len 1")


def test_super_init_mutates_class_var(nb_runner):
    _rerun(nb_runner,
           "class Base:\n    seen = []\n    def __init__(self):\n        Base.seen.append('x')\nclass Sub(Base):\n    def __init__(self):\n        super().__init__()",
           "s = Sub()\nprint('len', len(Base.seen))", "len 1")


def test_inherited_method_mutates_class_var(nb_runner):
    _rerun(nb_runner,
           "class Base:\n    log = []\n    def record(self):\n        Base.log.append(1)\nclass Sub(Base):\n    pass\ns = Sub()",
           "s.record()\nprint('len', len(Base.log))", "len 1")


def test_subclass_enter_override_mutates_class_var(nb_runner):
    _rerun(nb_runner,
           "class Base:\n    events = []\n    def __enter__(self):\n        return self\n    def __exit__(self, *a):\n        return False\nclass Sub(Base):\n    def __enter__(self):\n        Base.events.append('e')\n        return self",
           "with Sub() as s:\n    pass\nprint('len', len(Base.events))", "len 1")


def test_inherited_method_shared_attr_sibling(nb_runner):
    _rerun(nb_runner,
           "class Base:\n    shared = []\n    def add(self, v):\n        self.shared.append(v)\nclass Sub(Base):\n    pass\na = Sub()\nb = Sub()",
           "a.add('v')\nprint('len', len(b.shared))", "len 1")


# --- must NOT over-invalidate (pure) ------------------------------------------

def test_pure_inheritance_not_over_invalidated(nb_runner):
    _rerun(nb_runner,
           "class Base:\n    def __init__(self, n):\n        self.n = n\n    def doubled(self):\n        return self.n * 2\nclass Sub(Base):\n    def tripled(self):\n        return self.n * 3",
           "s = Sub(4)\nprint('r', s.doubled(), s.tripled())", "r 8 12")

"""Hidden-state mutation through a called function must reset on isolated re-run
(CAS-68). A cell calls a function that mutates state NOT passed as an argument
(a module global, a mutable default arg, a closure cell, a class variable, or a
function attribute); on isolated re-run the state accumulates because the
long-lived object created by an upstream cell is reused.

Each helper appends/increments ONCE, so the correct value is identical on the
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


# --- (A) global / free-variable mutation via a called function ---------------

def test_global_increment(nb_runner):
    _rerun(nb_runner, "g = 0\ndef bump():\n    global g\n    g += 1",
           "bump()\nprint('g', g)", "g 1")


def test_global_list_append(nb_runner):
    _rerun(nb_runner, "items = []\ndef add():\n    items.append(1)",
           "add()\nprint('len', len(items))", "len 1")


def test_global_dict_mutation(nb_runner):
    _rerun(nb_runner, "store = {}\ndef put():\n    store['k'] = store.get('k', 0) + 1",
           "put()\nprint('v', store['k'])", "v 1")


# --- (B) state on the function / class object itself -------------------------

def test_mutable_default_list(nb_runner):
    _rerun(nb_runner, "def collect(x, acc=[]):\n    acc.append(x)\n    return acc",
           "r = collect(1)\nprint('len', len(r))", "len 1")


def test_mutable_default_dict(nb_runner):
    _rerun(nb_runner, "def tally(k, acc={}):\n    acc[k] = acc.get(k, 0) + 1\n    return acc",
           "r = tally('a')\nprint('v', r['a'])", "v 1")


@pytest.mark.xfail(reason="CAS-68 part B: closure-cell state needs resolving the "
                          "factory producer of `c = make_counter()`", strict=False)
def test_closure_counter(nb_runner):
    _rerun(nb_runner,
           "def make_counter():\n    n = 0\n    def inc():\n        nonlocal n\n        n += 1\n        return n\n    return inc\nc = make_counter()",
           "print('c', c())", "c 1")


def test_class_variable_mutation(nb_runner):
    _rerun(nb_runner,
           "class Reg:\n    items = []\n    @classmethod\n    def add(cls, x):\n        cls.items.append(x)",
           "Reg.add(1)\nprint('len', len(Reg.items))", "len 1")


def test_function_attribute_counter(nb_runner):
    _rerun(nb_runner, "def tick():\n    tick.count = getattr(tick, 'count', 0) + 1\n    return tick.count",
           "print('t', tick())", "t 1")

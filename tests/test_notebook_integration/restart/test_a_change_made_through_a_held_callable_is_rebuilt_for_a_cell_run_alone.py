"""After a kernel restart, a cell run alone sees what the cells above changed through a callable they held.

``f = lst.append`` then ``f(4)``, ``list(map(lst.append, xs))``, a
``functools.partial``, ``getattr(lst, 'append')(4)``, a closure a factory
returned, a decorator that registers, a closure that counts its calls: in
the kernel that ran them cash reads what the call changes from the object.
After a restart the object is gone, and a cell run alone printed the list
without the items, silently, where Restart & Run All prints them.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

CASES = {
    "a_bound_method_kept_in_a_variable": (
        ["lst = [1, 2, 3]", "f = lst.append\nf(4)"],
        "print('R', lst)",
        "R [1, 2, 3, 4]",
    ),
    "a_bound_method_handed_to_map": (
        ["lst = [1, 2, 3]", "list(map(lst.append, [4, 5]))"],
        "print('R', lst)",
        "R [1, 2, 3, 4, 5]",
    ),
    "a_partial": (
        ["import functools\nlst = [1, 2, 3]", "p = functools.partial(lst.append, 4)\np()"],
        "print('R', lst)",
        "R [1, 2, 3, 4]",
    ),
    "a_method_looked_up_by_getattr": (
        ["lst = [1, 2, 3]", "getattr(lst, 'append')(4)"],
        "print('R', lst)",
        "R [1, 2, 3, 4]",
    ),
    "a_bound_method_handed_to_a_helper": (
        ["lst = [1]\ndef each(fn, xs):\n    for x in xs: fn(x)", "each(lst.append, [2, 3])"],
        "print('R', lst)",
        "R [1, 2, 3]",
    ),
    "a_closure_a_factory_returned": (
        ["lst = [1, 2, 3]\ndef mk():\n    def inner():\n        lst.append(4)\n    return inner", "g = mk()", "g()"],
        "print('R', lst)",
        "R [1, 2, 3, 4]",
    ),
    "a_decorator_that_registers": (
        [
            "registry = []\ndef register(f):\n    registry.append(f.__name__)\n    return f",
            "@register\ndef a(): pass",
            "@register\ndef b(): pass",
        ],
        "print('R', registry)",
        "R ['a', 'b']",
    ),
    "a_closure_that_counts_its_calls": (
        [
            "def make():\n    n = 0\n    def inc():\n        nonlocal n\n        n += 1\n        return n\n    return inc\ninc = make()",
            "a = inc()",
            "b = inc()",
        ],
        "c = inc()\nprint('R', c)",
        "R 3",
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_reader_run_alone_after_a_restart_sees_the_change(nb_runner, case):
    above, reader, expected = CASES[case]
    cells = [*above, reader]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expected in nb_runner.get_output(len(cells)), "control: the top-to-bottom run"

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cell(len(cells))

    assert expected in nb_runner.get_output(len(cells)), nb_runner.get_raw_output(len(cells))


def test_a_second_restart_still_sees_the_change(nb_runner):
    """The rebuild after the first restart runs the statement again, with
    the objects there: what it leaves on record is what the next kernel reads."""
    cells = ["lst = [1, 2, 3]", "f = lst.append\nf(4)", "print('R', lst)"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    for _ in range(2):
        nb_runner.restart()
        nb_runner._init_cash()
        nb_runner.run_cell(3)
        assert "R [1, 2, 3, 4]" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

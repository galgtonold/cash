"""On Restart & Run All, a call made through a callable kept in a variable happens again.

``w = buf.write`` then ``w("hi")``, ``nx = functools.partial(next, it)``
then ``a = nx()``, ``acq = lk.acquire`` then ``acq()``, a bare
``fmt.register(int, fi)`` on a ``functools.singledispatch`` function, ``del
a``: stored, the statement was a cache hit after the restart, and the
buffer stayed empty, the iterator was not advanced, the lock was not taken,
the registration was missing and ``a`` was bound again.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

CASES = {
    "a_bound_method_of_a_stream": (
        ["import io\nbuf = io.StringIO()\nw = buf.write", 'w("hi")'],
        "print('R', repr(buf.getvalue()))",
        "R 'hi'",
    ),
    "a_partial_of_print_to_a_stream": (
        ["import functools, io\nbuf = io.StringIO()\nlog = functools.partial(print, file=buf)", 'log("hi")'],
        "print('R', repr(buf.getvalue()))",
        "R 'hi\\n'",
    ),
    "a_registration_on_a_singledispatch_function": (
        [
            "import functools\n@functools.singledispatch\ndef fmt(x): return 'base'",
            'def fi(x): return "int"\nfmt.register(int, fi)',
        ],
        "print('R', fmt(1))",
        "R int",
    ),
    "a_partial_of_next": (
        ["import functools\nit = iter([1, 2, 3])\nnx = functools.partial(next, it)", "a = nx()"],
        "print('R', a, list(it))",
        "R 1 [2, 3]",
    ),
    "a_bound_method_of_a_lock": (
        ["import threading\nlk = threading.Lock()\nacq = lk.acquire", "acq()"],
        "print('R', lk.locked())",
        "R True",
    ),
    "a_del_of_a_name": (["a = 1", "del a"], "print('R', 'a' in globals())", "R False"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_call_happens_again_when_every_statement_is_stored(nb_runner, case):
    above, reader, expected = CASES[case]
    cells = [*above, reader]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert expected in nb_runner.get_output(len(cells)), "control: the first run"

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.enable_persist()
    nb_runner.run_all()

    assert expected in nb_runner.get_output(len(cells)), nb_runner.get_raw_output(len(cells))


def test_a_slow_generator_step_through_its_own_next_is_taken_again(nb_runner):
    """With the default settings: one step of the generator is slow enough
    for the statement to be stored."""
    cells = [
        "import time\ndef batches():\n    for i in range(3):\n        time.sleep(1.0)\n        yield i\nit = batches()\nnx = it.__next__",
        "a = nx()",
        "print('R', a, list(it))",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R 0 [1, 2]" in nb_runner.get_output(3), "control: the first run"

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()

    assert "R 0 [1, 2]" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

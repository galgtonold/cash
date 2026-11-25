"""A call made through a callable kept in a variable is judged as the same
call spelled out.

``buf.write('hi')``, ``next(it)`` and ``lk.acquire()`` are not stored: a
hit would not write to the stream, advance the iterator or take the lock.
Kept in a variable (``w = buf.write`` then ``w('hi')``, ``nx =
functools.partial(next, it)`` then ``a = nx()``), the same call named only
the variable and was stored: on Restart & Run All the buffer stayed empty
and the iterator where it was. So was a bare ``fmt.register(int, fi)`` on a
``functools.singledispatch`` function, and ``del a``.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell

SETUP = (
    "import functools, io, threading\n"
    "buf = io.StringIO()\nw = buf.write\nlog = functools.partial(print, file=buf)\n"
    "it = iter([1, 2, 3])\nnx = functools.partial(next, it)\nnx2 = it.__next__\n"
    "lk = threading.Lock()\nacq = lk.acquire\n"
    "@functools.singledispatch\ndef fmt(x):\n    return 'base'\n"
    "def fi(x):\n    return 'int'\n"
    "lst = [1]\nadd = lst.append\nhalf = functools.partial(round, ndigits=1)\na = 1"
)

REFUSED = {
    "a bound method of a stream": ("w('hi')", "iterator or stream"),
    "a partial handed a stream": ("log('hi')", "iterator or stream"),
    "a partial of next over an iterator": ("b = nx()", "iterator or stream"),
    "an iterator's own __next__": ("b = nx2()", "iterator or stream"),
    "a held callable handed to map": (
        "got = list(map(lambda _: nx2(), [0]))\nc = list(map(w, ['x']))",
        "iterator or stream",
    ),
    "a bound method of a lock": ("acq()", "a lock"),
    # Counted as changing the function, which also makes the statement a
    # producer of it for the cells that call the function.
    "a registration on a function": ("fmt.register(int, fi)", "In-place mutation on: fmt"),
    "a del of a name": ("del a", "Deletes 'a'"),
}

STORED = {
    "a bound method of a list": "n = add(2)",
    "a partial of a pure function": "h = half(2.26)",
    "a result read from a function": "impl = fmt.dispatch(int)",
    "a plain assignment": "z = a + 1",
}


@pytest.fixture
def reasons(cash_magics, cash_instance, statement_processor):
    """Runs a statement and returns why it was not stored (empty when it
    may be), with every statement cheap enough to store."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    run_cash_cell(cash_magics, SETUP)
    seen: dict[str, list[str]] = {}
    post_execute = statement_processor._post_execute

    def noting(run, execution):
        post_execute(run, execution)
        seen[run.code] = list(run.metrics.get("uncacheable_reasons") or ()) if run.skip_cache else []

    statement_processor._post_execute = noting

    def run(code: str) -> list[str]:
        run_cash_cell(cash_magics, code)
        return [reason for statement in code.split("\n") for reason in seen.get(statement, ["<not run>"])]

    return run


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_the_call_through_the_variable_is_not_stored(reasons, case):
    code, why = REFUSED[case]

    found = reasons(code)

    assert any(why in reason for reason in found), found


@pytest.mark.parametrize("case", sorted(STORED))
def test_a_callable_holding_nothing_a_hit_would_skip_is_still_stored(reasons, case):
    """Control: a bound method of a list (its receiver is restored with the
    statement) and a partial of a pure function are stored as before."""
    found = reasons(STORED[case])

    assert not any(
        "iterator or stream" in r or "a lock" in r or "on a function" in r or "Deletes" in r for r in found
    ), found

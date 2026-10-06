"""``@cash.cache(key=fn)``: what *fn* returns for a call replaces the call's
arguments in the key, and nothing else in the key moves.

*fn* gets the call bound to the cached function's signature with its
defaults applied, so every spelling of one call reaches it the same way. Its
own code is part of the key. If it raises, the call runs uncached with
KEY-FUNCTION-RAISED; if it reads something besides its arguments, cash says
so with KEY-FUNCTION-IMPURE. The checks that look at the arguments (did the
body change one in place?) still look at all of them.
"""

from __future__ import annotations

import asyncio
import textwrap
import time
import warnings

import pytest

from cash import CashImpureFunctionError
from tests._scripts import run_python


def _codes(caught) -> list[str]:
    return [getattr(w.message, "code", None) for w in caught]


def _unit(x, unit=1):
    return (x, int(unit))


def test_calls_the_key_function_maps_together_share_one_entry(disk_cash):
    @disk_cash.cache(key=_unit)
    def scale(x, unit=1):
        return x * int(unit)

    assert scale(2, "3") == 6
    assert scale(2, 3) == 6
    assert scale(2, unit="3") == 6
    assert scale(x=2, unit=3) == 6
    info = scale.cache_info()
    assert (info["misses"], info["hits"]) == (1, 3)
    assert scale(2, 4) == 8  # a different key still misses
    assert scale.cache_info()["misses"] == 2


def test_the_key_function_gets_the_call_bound_with_its_defaults(cash_instance):
    seen = []

    def key(x, unit=1, *, scale=10):
        seen.append((x, unit, scale))
        return (x, unit, scale)

    @cash_instance.cache(key=key)
    def f(x, unit=1, *, scale=10):
        return x * unit * scale

    f(1)
    f(x=1)
    f(1, 1, scale=10)
    assert seen == [(1, 1, 10)] * 3
    assert f.cache_info()["hits"] == 2


def test_the_rest_of_the_key_is_unchanged(cash_instance):
    """A global the body reads is still keyed, though the key function ignores it."""
    import types

    mod = types.ModuleType("keyed_globals_mod")
    exec(
        textwrap.dedent(
            """
            FACTOR = 2

            def f(x, unit):
                return x * FACTOR
            """
        ),
        mod.__dict__,
    )
    cached = cash_instance.cache(key=lambda x, unit: x)(mod.f)
    assert cached(3, "a") == 6
    assert cached(3, "b") == 6
    assert cached.cache_info()["hits"] == 1
    mod.FACTOR = 5
    assert cached(3, "a") == 15


def test_a_method_passes_self_to_the_key_function(cash_instance):
    class Model:
        def __init__(self, name):
            self.name = name

        @cash_instance.cache(key=lambda self, x, verbose=False: (self.name, x))
        def predict(self, x, verbose=False):
            return f"{self.name}:{x}"

    a = Model("a")
    assert a.predict(1) == "a:1"
    assert a.predict(1, verbose=True) == "a:1"
    assert Model("b").predict(1) == "b:1"
    info = Model.predict.cache_info()
    assert (info["hits"], info["misses"]) == (1, 2)


def test_an_async_function_is_keyed_by_its_key_function(cash_instance):
    @cash_instance.cache(key=lambda x, unit=1: (x, int(unit)))
    async def fetch(x, unit=1):
        await asyncio.sleep(0)
        return x * int(unit)

    async def main():
        return [await fetch(2, "3"), await fetch(2, 3)]

    assert asyncio.run(main()) == [6, 6]
    assert fetch.cache_info()["hits"] == 1


def test_a_generator_function_is_keyed_by_its_key_function(cash_instance):
    @cash_instance.cache(key=lambda n, label="": n)
    def count(n, label=""):
        yield from range(n)

    assert list(count(3, "first")) == [0, 1, 2]
    assert list(count(3, "second")) == [0, 1, 2]
    assert count.cache_info()["hits"] == 1


def test_a_key_function_that_raises_runs_the_call_uncached(cash_instance):
    runs = []

    def key(x):
        return {"a": 1}[x]

    @cash_instance.cache(key=key)
    def f(x):
        runs.append(x)  # @cash:assume-safe
        return x

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert f("b") == "b"
        assert f("b") == "b"
    assert runs == ["b", "b"]
    assert _codes(caught).count("KEY-FUNCTION-RAISED") == 1
    assert "KeyError" in str(next(w.message for w in caught if "KEY-FUNCTION-RAISED" in str(w.message)))
    assert f("a") == "a" and f("a") == "a"
    assert runs == ["b", "b", "a"]


def test_a_key_function_that_reads_the_clock_or_a_random_number_is_reported(cash_instance):
    import random

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @cash_instance.cache(key=lambda x: (x, time.time()))
        def stamped(x):
            return x

    assert "KEY-FUNCTION-IMPURE" in _codes(caught)
    assert "time.time()" in str(caught[-1].message)

    def drawn(x):
        return (x, random.random())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @cash_instance.cache(key=drawn)
        def sampled(x):
            return x

    assert "KEY-FUNCTION-IMPURE" in _codes(caught)
    assert "random.random()" in str(caught[-1].message)


def test_a_key_function_that_reads_a_file_is_reported(cash_instance, tmp_path):
    path = tmp_path / "units.txt"
    path.write_text("1", encoding="utf-8")

    def by_open(x):
        with open(path, encoding="utf-8") as fh:
            return (x, fh.read())

    def by_reader(x):
        return (x, path.read_text(encoding="utf-8"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @cash_instance.cache(key=by_open)
        def f(x):
            return x

    assert "KEY-FUNCTION-IMPURE" in _codes(caught)

    # A reader the code check does not name is caught on the first call.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @cash_instance.cache(key=by_reader)
        def g(x):
            return x

        assert "KEY-FUNCTION-IMPURE" not in _codes(caught)
        g(1)
    assert "KEY-FUNCTION-IMPURE" in _codes(caught)
    assert str(path) in str(next(w.message for w in caught if "KEY-FUNCTION-IMPURE" in str(w.message)))


def test_a_pure_key_function_gives_no_purity_warning(cash_instance):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @cash_instance.cache(key=_unit)
        def f(x, unit=1):
            return x

        f(1, "2")
    assert "KEY-FUNCTION-IMPURE" not in _codes(caught)


def test_strict_raises_on_an_impure_key_function_and_assume_safe_silences_it(cash_instance):
    with pytest.raises(CashImpureFunctionError, match="key= function"):

        @cash_instance.cache(key=lambda x: (x, time.time()), strict=True)
        def f(x):
            return x

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @cash_instance.cache(key=lambda x: (x, time.time()), assume_safe=True)
        def g(x):
            return x

    assert "KEY-FUNCTION-IMPURE" not in _codes(caught)


def test_the_argument_mutation_check_still_sees_every_argument(cash_instance):
    """The key function leaves ``rows`` out; the body changing it in place is
    still seen, and a body that leaves it alone is stored."""

    @cash_instance.cache(key=lambda rows, n: n)
    def grow(rows, n):
        rows.append(n)
        return len(rows)

    @cash_instance.cache(key=lambda rows, n: n)
    def size(rows, n):
        return len(rows) + n

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grow([1], 2)
        grow([1], 2)
    assert grow.cache_info()["hits"] == 0  # never stored: the call changed an argument

    size([1], 2)
    size([1, 2], 2)
    assert size.cache_info()["hits"] == 1


def test_key_must_be_a_plain_function(cash_instance):
    with pytest.raises(TypeError, match="key= takes a function"):
        cash_instance.cache(key="x")(lambda x: x)

    async def akey(x):
        return x

    with pytest.raises(TypeError, match="not an async or generator"):
        cash_instance.cache(key=akey)(lambda x: x)


def test_explain_says_when_a_hit_was_matched_by_the_key_function(cash_instance):
    @cash_instance.cache(key=_unit)
    def scale(x, unit=1):
        return x * int(unit)

    scale(2, "3")
    same = scale.explain(2, "3")
    assert same.would_hit and "matched_by" not in same.details
    other = scale.explain(2, 3)
    assert other.would_hit
    assert "key=" in other.details["matched_by"]
    assert "other arguments" in other.details["matched_by"]


_SCRIPT = """
import sys
import cash

{key_src}

@cash.cache(key=key)
def scale(x, unit=1):
    return x * int(unit)

scale(2, sys.argv[1])
info = scale.cache_info()
print("HITS", info["hits"], "MISSES", info["misses"])
"""


def _write(tmp_path, key_src: str) -> None:
    (tmp_path / "job.py").write_text(_SCRIPT.format(key_src=textwrap.dedent(key_src)), encoding="utf-8")


def test_the_key_is_stable_across_processes_and_an_edit_to_the_key_function_rekeys(tmp_path):
    _write(
        tmp_path,
        """
        def key(x, unit=1):
            return (x, int(unit))
        """,
    )
    first = run_python("job.py", "3", cwd=tmp_path)
    assert "HITS 0 MISSES 1" in first.stdout
    again = run_python("job.py", "3", cwd=tmp_path)
    assert "HITS 1 MISSES 0" in again.stdout, again.stdout + again.stderr
    as_text = run_python("job.py", " 3", cwd=tmp_path)  # int(" 3") == 3: the same key
    assert "HITS 1 MISSES 0" in as_text.stdout, as_text.stdout + as_text.stderr

    _write(
        tmp_path,
        """
        def key(x, unit=1):
            return (x, int(unit), "v2")
        """,
    )
    edited = run_python("job.py", "3", cwd=tmp_path, env={"CASH_VERBOSE": "1"})
    assert "HITS 0 MISSES 1" in edited.stdout, edited.stdout + edited.stderr
    assert "key= function key changed" in edited.stderr, edited.stderr


def test_an_edit_to_a_helper_of_the_key_function_rekeys(tmp_path):
    def write(helper_body: str) -> None:
        _write(
            tmp_path,
            f"""
            def _norm(unit):
                return {helper_body}

            def key(x, unit=1):
                return (x, _norm(unit))
            """,
        )

    write("int(unit)")
    run_python("job.py", "3", cwd=tmp_path)
    assert "HITS 1" in run_python("job.py", "3", cwd=tmp_path).stdout
    write("int(unit) + 0")
    assert "HITS 0 MISSES 1" in run_python("job.py", "3", cwd=tmp_path).stdout


def _by(field):
    return lambda rec: rec[field]


class _Pick:
    def __init__(self, field):
        self.field = field

    def __call__(self, rec):
        return rec[self.field]


@pytest.mark.parametrize("make", [_by, _Pick], ids=["factory closure", "callable object"])
def test_what_the_key_function_holds_is_part_of_the_key(cash_instance, make):
    """``key=by("id")`` edited to ``key=by("sku")`` shares the key function's
    code; only the captured field (or the object's state) differs. Keyed by
    code alone, the record stored under id 1 was served for sku 1."""

    def price(rec):
        return rec["price"]

    assert cash_instance.cache(key=make("id"))(price)({"id": 1, "sku": 5, "price": 10}) == 10
    by_sku = cash_instance.cache(key=make("sku"))(price)
    assert by_sku({"id": 7, "sku": 1, "price": 99}) == 99, "an entry keyed by id was served for a sku"


def test_a_key_function_holding_what_cannot_be_hashed_runs_uncached(cash_instance):
    class Conn:
        offset = 0

        def __reduce__(self):
            raise TypeError("a live connection")

    conn = Conn()

    @cash_instance.cache(key=lambda x: x + conn.offset)
    def f(x):
        return x

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert f(1) == 1
        assert f(1) == 1
    assert "KEY-UNHASHABLE-CAPTURE" in _codes(caught)
    assert f.cache_info()["hits"] == 0


def test_a_key_function_that_consumes_an_iterator_does_not_store_the_result(cash_instance):
    """``key=lambda rows: tuple(rows)`` empties a generator before the body
    runs. The body's sum of nothing was stored under (1, 2, 3) and served
    later even for a list of those rows."""

    @cash_instance.cache(key=lambda rows: tuple(rows))
    def total(rows):
        return sum(rows)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        total(r for r in [1, 2, 3])
    assert "KEY-ITERATOR-CONSUMED" in _codes(caught)
    assert total([1, 2, 3]) == 6, "the result computed from the emptied generator was stored"


def test_a_registered_hasher_that_consumes_an_iterator_does_not_store_the_result(cash_instance):
    import types

    cash_instance.register_hasher(types.GeneratorType, lambda g: repr(tuple(g)))

    @cash_instance.cache
    def total(rows):
        return sum(rows)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        total(r for r in [1, 2, 3])
        assert total(r for r in [1, 2, 3]) == 0  # the body saw the emptied generator, as before
    assert "KEY-ITERATOR-CONSUMED" in _codes(caught)
    assert total.cache_info()["hits"] == 0


def test_an_iterator_the_key_function_does_not_read_is_cached(cash_instance):
    """The control: the iterator is left as it was, and the call caches."""

    @cash_instance.cache(key=lambda rows, n: n)
    def twice(rows, n):
        return 2 * n

    gen = (r for r in [1, 2, 3])
    assert twice(gen, 2) == 4
    assert twice(iter([1]), 2) == 4 and twice.cache_info()["hits"] == 1
    assert list(gen) == [1, 2, 3]


def test_explain_leaves_an_iterator_argument_unread(cash_instance):
    @cash_instance.cache(key=lambda rows: tuple(rows))
    def total(rows):
        return sum(rows)

    rows = (r for r in [1, 2, 3])
    answer = total.explain(rows)
    assert answer.details.get("error") == "KEY-ITERATOR-CONSUMED", answer
    assert list(rows) == [1, 2, 3], "explain() emptied the caller's generator"

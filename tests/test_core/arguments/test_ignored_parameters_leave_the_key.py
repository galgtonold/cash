"""``@cash.cache(ignore=[...])`` and ``cash.Ignore``: the named parameters
are left out of the argument part of the key, and nothing else is.

The names are checked against the signature when the function is decorated.
The list and the annotation make one set; either combined with ``key=``
raises. The checks that look at the arguments still see the ignored ones.
"""

from __future__ import annotations

import logging
import textwrap
import warnings
from typing import Annotated

import pytest

import cash
from tests._scripts import run_python


def test_an_ignored_parameter_does_not_split_entries(cash_instance):
    @cash_instance.cache(ignore=["debug"])
    def load(path, debug=False):
        return path.upper()

    assert load("a") == "A"
    assert load("a", debug=True) == "A"
    assert load("a", True) == "A"
    assert load(path="a", debug=False) == "A"
    info = load.cache_info()
    assert (info["misses"], info["hits"]) == (1, 3)
    assert load("b", debug=True) == "B"  # the other arguments still key
    assert load.cache_info()["misses"] == 2


def test_a_bare_name_is_one_parameter(cash_instance):
    @cash_instance.cache(ignore="verbose")
    def f(x, verbose=0):
        return x

    f(1, verbose=1)
    f(1, verbose=2)
    assert f.cache_info()["hits"] == 1


def test_a_name_that_is_not_a_parameter_raises_and_lists_the_real_ones(cash_instance):
    def load(path, debug=False):
        return path

    with pytest.raises(ValueError, match=r"no parameter 'debgu' in load\(path, debug\)"):
        cash_instance.cache(ignore=["debgu"])(load)

    with pytest.raises(TypeError, match="parameter names as strings"):
        cash_instance.cache(ignore=[1])(load)


def test_a_library_function_can_ignore_a_parameter(cash_instance):
    """``cash.cache(ignore=[...])(fn)`` for a function you do not own."""
    import textwrap as lib

    cached = cash_instance.cache(ignore=["width"])(lib.fill)
    assert cached("a b c", width=10) == "a b c"
    assert cached("a b c", width=20) == "a b c"  # served: width is not in the key
    assert cached.cache_info()["hits"] == 1


@pytest.mark.parametrize("spelling", ["subscript", "annotated"])
def test_an_annotated_parameter_is_ignored(cash_instance, spelling):
    if spelling == "subscript":

        @cash_instance.cache
        def f(x, debug: cash.Ignore[bool] = False):
            return x

    else:

        @cash_instance.cache
        def f(x, debug: Annotated[bool, cash.Ignore] = False):
            return x

    f(1)
    f(1, debug=True)
    assert f.cache_info()["hits"] == 1


def test_a_type_checker_sees_the_wrapped_type():
    import typing

    assert typing.get_args(cash.Ignore[bool])[0] is bool


def test_the_list_and_the_annotation_make_one_set(cash_instance):
    @cash_instance.cache(ignore=["log", "debug"])
    def f(x, debug: cash.Ignore[bool] = False, log=None):
        return x

    f(1)
    f(1, debug=True, log=logging.getLogger("x"))
    assert f.cache_info()["hits"] == 1


def test_key_and_ignore_together_raise(cash_instance):
    def f(x, debug: cash.Ignore[bool] = False):
        return x

    def g(x, debug=False):
        return x

    with pytest.raises(TypeError, match="key= cannot be combined with ignored parameters"):
        cash_instance.cache(key=lambda x, debug=False: x)(f)
    with pytest.raises(TypeError, match="key= cannot be combined with ignored parameters"):
        cash_instance.cache(key=lambda x, debug=False: x, ignore=["debug"])(g)


def test_ignoring_star_args_and_star_star_kwargs(cash_instance):
    @cash_instance.cache(ignore=["extra", "options"])
    def f(x, *extra, **options):
        return x

    f(1, 2, 3, a=1)
    f(1, b=2)
    assert f.cache_info()["hits"] == 1

    @cash_instance.cache
    def g(x, *extra: cash.Ignore[int]):
        return x

    g(1, 2)
    g(1, 3)
    assert g.cache_info()["hits"] == 1


def test_an_unhashable_ignored_argument_does_not_stop_caching(cash_instance):
    class Unpicklable:
        def __reduce__(self):
            raise TypeError("no")

    @cash_instance.cache(ignore=["handle"])
    def f(x, handle=None):
        return x

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f(1, Unpicklable())
        f(1, Unpicklable())
    assert f.cache_info()["hits"] == 1
    assert "KEY-UNHASHABLE-ARG" not in [getattr(w.message, "code", None) for w in caught]


class _Connection:
    def __reduce__(self):
        raise TypeError("a live connection")


_CONN = _Connection()


@pytest.mark.parametrize("how", ["ignore=", "cash.Ignore", "key="])
def test_an_unhashable_default_of_a_parameter_left_out_does_not_stop_caching(cash_instance, how):
    """A parameter left out of the key is left out with its default: a
    connection default made the function never cache, with
    KEY-UNHASHABLE-DEFAULT."""
    if how == "ignore=":

        @cash_instance.cache(ignore=["conn"])
        def f(table, conn=_CONN):
            return table.upper()

    elif how == "cash.Ignore":

        @cash_instance.cache
        def f(table, conn: cash.Ignore[_Connection] = _CONN):
            return table.upper()

    else:

        @cash_instance.cache(key=lambda table, conn: table)
        def f(table, conn=_CONN):
            return table.upper()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            assert f("users") == "USERS"
    assert f.cache_info()["hits"] == 2
    assert "KEY-UNHASHABLE-DEFAULT" not in [getattr(w.message, "code", None) for w in caught]


def test_the_default_of_a_keyed_parameter_still_stops_caching(cash_instance):
    """The control: the same default on a parameter that is keyed."""

    @cash_instance.cache(ignore=["verbose"])
    def f(table, conn=_CONN, verbose=False):
        return table.upper()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f("users")
        f("users")
    assert f.cache_info()["hits"] == 0
    assert "KEY-UNHASHABLE-DEFAULT" in [getattr(w.message, "code", None) for w in caught]


def test_the_argument_mutation_check_still_sees_an_ignored_argument(cash_instance):
    @cash_instance.cache(ignore=["log"])
    def run(x, log):
        log.append(x)
        return x

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run(1, [])
        run(1, [])
    assert run.cache_info()["hits"] == 0  # the call changed an argument: never stored


def test_a_hit_does_not_run_the_body_so_a_debug_print_does_not_happen(cash_instance, capsys):
    @cash_instance.cache
    def f(x, debug: cash.Ignore[bool] = False):
        if debug:
            print("computing", x)  # @cash:assume-safe
        return x

    f(1)
    f(1, debug=True)
    assert capsys.readouterr().out == ""


def test_explain_says_when_a_hit_was_matched_by_ignored_parameters(cash_instance):
    @cash_instance.cache(ignore=["debug"])
    def f(x, debug=False):
        return x

    f(1)
    assert "matched_by" not in f.explain(1).details
    other = f.explain(1, debug=True)
    assert other.would_hit
    assert "ignored parameters (debug)" in other.details["matched_by"]


class _Model:
    """At module level: an instance of a class defined in a test cannot be pickled, so not keyed."""

    def fit(self, n, verbose=False):
        return n * 2


def test_a_method_can_ignore_a_parameter(cash_instance, monkeypatch):
    monkeypatch.setattr(_Model, "fit", cash_instance.cache(ignore=["verbose"])(_Model.__dict__["fit"]))
    m = _Model()
    m.fit(2)
    m.fit(2, verbose=True)
    assert _Model.fit.cache_info()["hits"] == 1


_FUTURE = """
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import cash

if TYPE_CHECKING:
    from decimal import Decimal


@cash.cache
def scale(x: Decimal, debug: cash.Ignore[bool] = False):
    return x

scale(2, debug=sys.argv[1] == "1")
info = scale.cache_info()
print("HITS", info["hits"], "MISSES", info["misses"])
"""


def test_string_annotations_are_read_and_the_key_holds_across_processes(tmp_path):
    """``from __future__ import annotations`` with a name only imported for
    type checkers: the hints cannot all be resolved, the Ignore one still is."""
    (tmp_path / "job.py").write_text(_FUTURE, encoding="utf-8")
    assert "HITS 0 MISSES 1" in run_python("job.py", "0", cwd=tmp_path).stdout
    again = run_python("job.py", "1", cwd=tmp_path)
    assert "HITS 1 MISSES 0" in again.stdout, again.stdout + again.stderr


def test_an_unreadable_annotation_that_names_ignore_raises_when_decorated(tmp_path):
    src = textwrap.dedent(
        """
        from __future__ import annotations
        import cash

        try:
            @cash.cache
            def f(x, debug: Missing[cash.Ignore[bool]] = False):
                return x
        except TypeError as e:
            print("RAISED", e)
        """
    )
    (tmp_path / "job.py").write_text(src, encoding="utf-8")
    out = run_python("job.py", cwd=tmp_path).stdout
    assert "RAISED" in out and "'debug'" in out and "ignore=['debug']" in out

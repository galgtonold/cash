"""A callee whose attributes refuse to be true or false is still keyed.

``pl.col`` answers any attribute with an expression (``pl.col.x`` is the
column ``x``), and an expression raises when asked whether it is true. The
helper walk asked ``if getattr(callee, "_cash_cached", False)``, which raised;
the failure was once taken for "no helpers" and later for "cannot key", so a
function calling ``pl.col(...)`` ran uncached. Only ``True`` itself marks a
cached function.
"""

from __future__ import annotations

import warnings

from cash import Cash


class _Ambiguous:
    def __bool__(self):
        raise TypeError("the truth value of an expression is ambiguous")


class _Columns:
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Ambiguous()

    def __call__(self, value):
        return value * 2


col = _Columns()


def _double(x):
    return col(x)


def test_a_function_calling_it_is_cached_without_a_warning():
    c = Cash()
    double = c.cache(_double)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert double(3) == 6
        assert double(3) == 6
    codes = [getattr(w.message, "code", None) for w in caught]
    assert "KEY-HELPERS-UNWALKABLE" not in codes
    assert [call["cache_hit"] for call in c.drain_decorator_calls()] == [False, True]


_LIB = """
class _Ambiguous:
    def __bool__(self):
        raise TypeError("the truth value of an expression is ambiguous")


class _Columns:
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Ambiguous()

    def __call__(self, value):
        return value


col = _Columns()


def make(n):
    return col(n) * FACTOR
"""


def test_editing_a_helper_that_calls_it_recomputes_in_a_new_process(tmp_path):
    # Seen on 0.11 with polars: the walk's failure was taken for "no helpers",
    # so editing the helper in another module kept serving the old result.
    from tests._scripts import run_python

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "study.py").write_text(
        "import cash\nfrom pkg.lib import make\n\n\n@cash.cache\ndef cached(n):\n    return make(n)\n",
        encoding="utf-8",
    )
    (tmp_path / "run.py").write_text("from pkg.study import cached\nprint(cached(1))\n", encoding="utf-8")
    lib = pkg / "lib.py"
    lib.write_text("FACTOR = 2\n" + _LIB, encoding="utf-8")
    assert run_python("run.py", cwd=tmp_path).stdout.split() == ["2"]
    lib.write_text("FACTOR = 3\n" + _LIB, encoding="utf-8")
    assert run_python("run.py", cwd=tmp_path).stdout.split() == ["3"]

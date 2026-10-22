"""What in a statement's text names a memo's counters (``cache_info``,
``cache_clear``), and so makes it rerun while a memo of the user's is about:
an attribute, a name, a string that is one wherever it stands, a string a
name is built from, a name another statement bound to one, and a
``getattr`` of a memo whose name another statement bound. Not a string
that only contains one: a path, a dict key, a sentence in an f-string ran
every time, with no counter read."""

from __future__ import annotations

import functools

import pytest

from cash.notebook.helper_memos import _text_names_counters


def _ns():
    # Made per test: a memo at the top of a test module is one of the
    # user's to every other test in the process.
    @functools.cache
    def f(n):
        return n

    return {"f": f, "n": "cache_info", "m": "cache_dir", "g": len}


@pytest.mark.parametrize(
    "code",
    [
        "a = f.cache_info()",
        "c = cache_clear",
        "a = getattr(f, 'cache_info')()",
        "a = getattr(f, 'cache_' + 'info')()",
        "a = getattr(f, f'cache_{k}')()",
        "a = getattr(f, 'cache_{}'.format(k))()",
        "a = operator.methodcaller('cache_info')(f)",
        "a = getattr(f, n)()",
        # The exact string, outside getattr.
        "a = f.__getattribute__('cache_info')()",
        "a = object.__getattribute__(f, 'cache_info')()",
        "a = vars(type(f))['cache_info'](f)",
        "a = type(f).__dict__['cache_info'](f)",
        "a = functools.reduce(getattr, ['cache_info'], f)()",
        "a = h(f, 'cache_info')()",
        # A name bound to one in another statement, whatever reads it.
        "a = info(f, n)",
        "a = h(f, n)()",
        "a = getattr(fs[0], n)()",
        "a = getattr(box.f, n)()",
    ],
)
def test_a_counter_named_reruns(code):
    assert _text_names_counters(code, _ns())


@pytest.mark.parametrize(
    "code",
    [
        "rows = read('cache_dir/rows.csv')",
        "rows = {'cache_info': 1}",
        "msg = f'see cache_info {1}'",
        "cache_info_rows = 3",
        "p = 'cache_dir/' + name",
        "a = getattr(g, m)",
        "a = getattr(obj, m)",
    ],
)
def test_a_string_that_only_contains_one_does_not(code):
    assert not _text_names_counters(code, _ns())

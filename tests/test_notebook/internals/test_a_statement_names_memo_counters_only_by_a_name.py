"""What in a statement's text names a memo's counters (``cache_info``,
``cache_clear``), and so makes it rerun while a memo of the user's is about:
an attribute, a name, a string that is one wherever it stands, a string a
name is built from, a name another statement bound to one, and a
``getattr`` of a memo whose name another statement bound. Not a string
that only contains one: a path, a dict key, a sentence in an f-string ran
every time, with no counter read."""

from __future__ import annotations

import functools
import types

import pytest

from cash.notebook.helper_memos import _names_clear, _text_names_counters


def _ns():
    # Made per test: a memo at the top of a test module is one of the
    # user's to every other test in the process.
    @functools.cache
    def f(n):
        return n

    return {
        "f": f,
        "n": "cache_info",
        "m": "cache_dir",
        "g": len,
        "names": ["cache_info"],
        "NAMES": {"i": "cache_info"},
        "cfg": types.SimpleNamespace(k="cache_info"),
        "mod": types.SimpleNamespace(K="cache_info"),
    }


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
        # A word the name is built from, however.
        "a = info(f, ''.join(['cache_', 'info']))",
        "a = info(f, 'cache_info_'[:-1])",
        "a = getattr(fs[0], ''.join(['cache_', 'info']))()",
        # Held one step down in what the statement names.
        "a = info(f, names[0])",
        "a = info(f, NAMES['i'])",
        "a = info(f, cfg.k)",
        "a = info(f, mod.K)",
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


K = "cache_info"
CFG = {"stat": "cache_info"}


def _by_global(g):
    return getattr(g, K)()


def _by_global_dict(g):
    return getattr(g, CFG["stat"])()


def _by_default(g, k="cache_info"):
    return getattr(g, k)()


def _by_keyword_default(g, *, k="cache_clear"):
    return getattr(g, k)()


def _by_closure():
    k = "cache_info"

    def info(g):
        return getattr(g, k)()

    return info


@pytest.mark.parametrize(
    "fn",
    [_by_global, _by_global_dict, _by_default, _by_keyword_default, _by_closure()],
    ids=["global", "global_dict", "default", "keyword_default", "closure"],
)
def test_a_helper_that_holds_a_counter_name_clears(fn):
    assert _names_clear(fn)


def _plain(g, k="cache_dir"):
    return getattr(g, k)


def test_a_helper_holding_other_words_does_not():
    assert not _names_clear(_plain)

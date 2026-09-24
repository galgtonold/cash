"""A call keyed on what it receives (``by_content``), and when it may be.

The integration arm is
``test_notebook_integration/calls/test_a_call_keys_on_what_it_receives.py``; this
pins the key's two changes and each reason :func:`_keys_by_content` says no.
"""

import ast

import pytest

from cash.notebook.cache_key import CacheKeyContext
from cash.notebook.call_interception import wrap_eligible_calls
from cash.notebook.call_key import (
    _CONTENT_KEY_MAX_BYTES,
    _NAME_CONTENT_MAX_BYTES,
    CallKeys,
    _keys_by_content,
    call_cache_key,
)

np = pytest.importorskip("numpy")

SWEEP = "sc = {mid: fit_score(make_features(cleaned[mid], W)) for mid in ids}"


def _site(stmt=SWEEP):
    _, sites = wrap_eligible_calls(ast.parse(stmt))
    return sites[0]


def _ctx(cleaned_lineage):
    lineage = {"cleaned": cleaned_lineage, "W": "w-3", "fit_score": "f", "make_features": "m"}
    return CacheKeyContext(variable_lineage=lineage, user_ns={})


def _key(site, ctx, by_content):
    return call_cache_key(site, ctx=ctx, arg_digests=["features"], loop_vars={"0:W": 3}, by_content=by_content)


def _fn(source, **namespace):
    exec(compile(source, "<cell>", "exec"), namespace)
    return namespace["f"]


# -- the key ------------------------------------------------------------------


def test_the_site_knows_which_names_only_feed_an_argument():
    assert _site().content_names == {"make_features", "cleaned", "W"}


def test_a_name_that_only_feeds_an_argument_leaves_the_key():
    site = _site()
    assert _key(site, _ctx("before"), True) == _key(site, _ctx("after-a-fix"), True)
    assert _key(site, _ctx("before"), False) != _key(site, _ctx("after-a-fix"), False)


BY_NAME = "models = {key: fit_series(g, PARAMS, cutoff) for key, g in groups}"


def _by_name_key(cutoff_lineage, cutoff_value, params=None):
    site = _site(BY_NAME)
    ctx = CacheKeyContext(
        variable_lineage={"PARAMS": "p", "cutoff": cutoff_lineage, "fit_series": "f", "groups": "g"}, user_ns={}
    )
    args = (None, params or {"alpha": 0.5}, cutoff_value)
    return call_cache_key(
        site,
        ctx=ctx,
        arg_digests=["group"],
        loop_vars={},
        by_content=True,
        name_digests=CallKeys._name_digests(site, args, {}),
    )


def test_the_site_knows_which_arguments_are_passed_by_name():
    assert dict(_site(BY_NAME).name_arg_positions) == {"PARAMS": 1, "cutoff": 2}
    # read elsewhere in the call too: not only an argument
    assert _site("x = f(a, a + 1)").name_arg_positions == ()


def test_an_argument_passed_by_name_is_keyed_on_its_value():
    # `cutoff` is rebuilt from the frame, so its lineage moves on any fix
    assert _by_name_key("before", 6) == _by_name_key("after-a-fix", 6)
    assert _by_name_key("x", 6) != _by_name_key("x", 7)
    assert _by_name_key("x", 6) != _by_name_key("x", 6, params={"alpha": 1.0})


def test_a_big_argument_passed_by_name_keeps_its_lineage():
    site = _site("x = f(big)")
    big = np.zeros(_NAME_CONTENT_MAX_BYTES // 8 + 1)
    assert CallKeys._name_digests(site, (big,), {}) == {}
    assert CallKeys._name_digests(site, (big[:10],), {}).keys() == {"big"}


def test_the_statement_leaves_the_key():
    renamed = _site(SWEEP.replace("sc =", "scores ="))
    assert renamed.stmt_identity != _site().stmt_identity
    assert _key(_site(), _ctx("x"), True) == _key(renamed, _ctx("x"), True)
    assert _key(_site(), _ctx("x"), False) != _key(renamed, _ctx("x"), False)


def test_the_argument_value_still_decides():
    site = _site()
    a = call_cache_key(site, ctx=_ctx("x"), arg_digests=["one"], loop_vars={}, by_content=True)
    b = call_cache_key(site, ctx=_ctx("x"), arg_digests=["two"], loop_vars={}, by_content=True)
    assert a != b


def test_a_key_by_content_never_equals_one_by_statement():
    site = _site("x = f(a + 1)")
    ctx = CacheKeyContext(variable_lineage={"a": "l", "f": "g"}, user_ns={})
    assert call_cache_key(site, ctx=ctx, arg_digests=["d"], loop_vars={}, by_content=True) != call_cache_key(
        site, ctx=ctx, arg_digests=["d"], loop_vars={}, by_content=False
    )


# -- when it may be -------------------------------------------------------------

PURE = """
import math
SCALE = 2.0
def helper(x):
    return x * SCALE
def f(frame):
    return math.fsum(helper(frame).ravel())
"""


def _decide(fn, args, loop_vars=None, stmt="y = f(g(a))"):
    return _keys_by_content(fn, _site(stmt), args, {}, loop_vars or {})


def test_plain_arguments_and_a_pure_callee_are_keyed_by_content():
    assert _decide(_fn(PURE), (np.arange(10.0),), {"0:W": 3})


def test_an_attribute_named_like_a_global_is_not_read_as_that_global():
    """``os.open`` loads the attribute ``open``, not the global -- which in a
    notebook is cash's file-tracking wrapper."""
    fn = _fn("import os\ndef f(x):\n    fd = os.open('p', os.O_RDONLY)\n    return x", open=lambda *a: None)
    assert _decide(fn, (1,))


@pytest.mark.parametrize(
    "arg",
    [
        object(),
        iter([1, 2]),
        (i for i in range(2)),
        np.array([object()]),
        [1, object()],
    ],
    ids=["object", "iterator", "generator", "object_array", "list_holding_one"],
)
def test_an_argument_whose_state_its_hash_may_miss_keeps_the_statement(arg):
    assert not _decide(_fn(PURE), (arg,))


def test_a_loop_variable_whose_state_its_hash_may_miss_keeps_the_statement():
    assert not _decide(_fn(PURE), (1,), {"0:conn": object()})


def test_a_dunder_loop_entry_is_not_asked():
    assert _decide(_fn(PURE), (1,), {"0:__iterable_lineage__": object()})


@pytest.mark.parametrize(
    "source",
    [
        "conn = object()\ndef f(x):\n    return (conn, x)",
        "conn = object()\ndef helper():\n    return conn\ndef f(x):\n    return helper()",
        "def make():\n    conn = object()\n    def f(x):\n        return (conn, x)\n    return f\nf = make()",
        "def f(x, cache=[object()]):\n    return x",
    ],
    ids=["global", "through_a_helper", "closure", "default"],
)
def test_a_callee_reaching_such_state_keeps_the_statement(source):
    """The cross-statement collision's ``fetch_next(conn)``, reached through the callee instead of
    an argument: two statements must keep their own entries."""
    assert not _decide(_fn(source), (1,))


def test_an_argument_too_big_to_hash_in_full_keeps_the_old_key():
    big = np.zeros(_CONTENT_KEY_MAX_BYTES // 8 + 1)
    assert not _decide(_fn(PURE), (big,))

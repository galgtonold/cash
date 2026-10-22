"""Review repro (5): the call-entry path of claude/served-memo-fill.

A sub-statement call entry (CallUnit) records the memos its callee can
reach *by name* (`helper_memos.watched_by`: the callee's co_names, nested
scopes, and its own name). Served, the call's body does not run, so a
memo it would have filled stays as it was.
"""

from __future__ import annotations

from cash.notebook.call_interception import CallSite

SRC = """
import functools, time
log = []
@functools.cache
def h(n):
    log.append(n)
    return n * 10
def g(n):
    return h(n)
def direct(n):
    time.sleep(0.05)
    return h(n) + 1
def via_helper(n):
    time.sleep(0.05)
    return g(n) + 1
"""


def _site(name):
    return CallSite(source=f"{name}(x)", free_names=frozenset({name, "x"}), occurrence_index=0)


def _ns():
    ns: dict = {"x": 5}
    exec(SRC, ns)
    return ns


def test_direct_memo_call_reruns_when_the_memo_was_emptied(call_unit_harness):
    ns = _ns()
    unit = call_unit_harness(lineage={"x": "hash-5"}, user_ns=ns)
    assert unit.wrap(ns["direct"], _site("direct"))(5) == 51
    ns["h"].cache_clear()  # as a restart leaves it
    assert unit.wrap(ns["direct"], _site("direct"))(5) == 51
    assert ns["h"].cache_info().currsize == 1, "plain Python leaves h(5) in the memo"
    assert [e["cache_hit"] for e in unit.call_log] == [False, False]


def test_direct_memo_call_still_serves_while_filled(call_unit_harness):
    # The first call fills h, so it is not stored (served, h would stay as
    # it was); the second only reads h, and the third is served over it.
    ns = _ns()
    unit = call_unit_harness(lineage={"x": "hash-5"}, user_ns=ns)
    for _ in range(3):
        unit.wrap(ns["direct"], _site("direct"))(5)
    assert [e["cache_hit"] for e in unit.call_log] == [False, False, True]


def test_memo_reached_through_another_notebook_function(call_unit_harness):
    # via_helper -> g -> h: h is not among via_helper's names, so the entry
    # records no memo and is served with h left empty.
    ns = _ns()
    unit = call_unit_harness(lineage={"x": "hash-5"}, user_ns=ns)
    assert unit.wrap(ns["via_helper"], _site("via_helper"))(5) == 51
    ns["h"].cache_clear()
    assert unit.wrap(ns["via_helper"], _site("via_helper"))(5) == 51
    assert ns["h"].cache_info().currsize == 1, (
        f"served with h empty: plain Python leaves h(5) in the memo; log={ns['log']}, hits={[e['cache_hit'] for e in unit.call_log]}"
    )

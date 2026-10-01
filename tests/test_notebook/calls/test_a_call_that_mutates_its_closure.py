"""A cached call whose closure holds an object it changes in place.

``add = make_log(); n1 = add('a'); n2 = add('a')`` with ``add`` appending to
a list ``seen`` held in its closure read ``n1 = 1, n2 = 1``: the list is in a
cell, not a global, so the key never saw it, and the second call was served
the first call's length without the append. Such a list is now treated as a
global the callee writes: its state before the call is in the key, its state
after is stored with the entry, and a hit puts it back into the live list.
"""

from __future__ import annotations

import time

from cash.notebook.call_interception import CallSite


def make_log():
    seen = []

    def add(x):
        time.sleep(0.05)  # above the call cost floor, so a miss is stored
        seen.append(x)
        return len(seen)

    return add, seen


def _site(stmt: str) -> CallSite:
    return CallSite(
        source="add('a')",
        free_names=frozenset({"add"}),
        occurrence_index=0,
        stmt_identity=stmt,
    )


def _run(unit, add):
    """``n1 = add('a')`` then ``n2 = add('a')``, as two statements."""
    unit.begin_statement()
    n1 = unit.wrap(add, _site("n1 = add('a')"))("a")
    unit.begin_statement()
    n2 = unit.wrap(add, _site("n2 = add('a')"))("a")
    return n1, n2


def test_a_second_call_on_the_changed_list_is_not_served_the_first(call_unit_harness):
    add, seen = make_log()
    unit = call_unit_harness(lineage={"add": "hash-add"}, user_ns={"add": add})

    unit.begin_statement()
    first = unit.wrap(add, _site("n = add('a')"))("a")
    unit.begin_statement()
    second = unit.wrap(add, _site("n = add('a')"))("a")

    assert (first, second) == (1, 2), "the second call was served the first call's length"
    assert seen == ["a", "a"]


def test_a_hit_puts_the_append_back_into_the_live_list(call_unit_harness):
    """The notebook run again: a new ``add`` over a new, empty list, with the
    same lineage. Both calls are served, and the list every holder sees is
    the one the calls built."""
    unit = call_unit_harness(lineage={"add": "hash-add"}, user_ns={})
    add, _ = make_log()
    assert _run(unit, add) == (1, 2)
    unit.drain()

    add, seen = make_log()
    held = seen

    assert _run(unit, add) == (1, 2)
    assert [e["cache_hit"] for e in unit.drain()] == [True, True], "the calls were run again"
    assert held == ["a", "a"], "the hit did not put the appends back into the live list"
    assert seen is held


def test_an_object_that_cannot_be_put_back_in_place_is_run_every_time(call_unit_harness):
    """A user object's state cannot be copied back into it soundly, so a call
    changing one in its closure is not cached."""

    class Tally:
        def __init__(self):
            self.n = 0

    def make_tally():
        tally = Tally()

        def bump():
            time.sleep(0.05)
            tally.n += 1
            return tally.n

        return bump

    bump = make_tally()
    unit = call_unit_harness(lineage={"bump": "hash-bump"}, user_ns={"bump": bump})
    site = CallSite(source="bump()", free_names=frozenset({"bump"}), occurrence_index=0)

    unit.begin_statement()
    first = unit.wrap(bump, site)()
    unit.begin_statement()
    second = unit.wrap(bump, site)()

    assert (first, second) == (1, 2)


def test_a_factory_is_not_served_the_closure_the_last_run_made(call_unit_harness):
    """A function is kept by reference, so a hit on ``make_log()`` handed
    back the closure the last run made, still holding the last run's list."""

    def factory():
        time.sleep(0.05)
        return make_log()

    unit = call_unit_harness(lineage={"make_log": "hash-make-log"}, user_ns={})
    site = CallSite(source="make_log()", free_names=frozenset({"make_log"}), occurrence_index=0)

    unit.begin_statement()
    first_add, first_seen = unit.wrap(factory, site)()
    first_add("a")
    unit.begin_statement()
    add, seen = unit.wrap(factory, site)()

    assert add is not first_add, "the closure the last run made was served again"
    assert (add("a"), seen) == (1, ["a"])

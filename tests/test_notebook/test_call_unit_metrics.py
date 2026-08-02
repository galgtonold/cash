"""Call-unit events reach the badge through the existing contract (CAS-243).

``CallUnit`` used to rely on ``CallCache`` rebuilding a ``module.qualname`` via
``Cash._get_func_key`` and stashing it in ``wrapped_names`` so the processor
could reconcile a drained decorator-log entry against "something this instance
actually wrapped" -- two independent renderings of the same name that had to
agree exactly, or the badge silently stopped marking intercepted calls. Each
event now sets ``intercepted=True`` at the point it is recorded, which is
strictly true by construction (every event ``CallUnit`` emits came from its
own interception), so there is nothing left to reconcile.
"""
from __future__ import annotations

import time

import pytest

from cash.notebook.badge_renderer.view_builder import build_interactive_badge
from cash.notebook.badge_renderer.renderers.text import render_text
from cash.notebook.call_interception import CallSite
from cash.notebook.call_unit import CallUnit
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


def test_call_unit_events_are_marked_intercepted_without_a_name_lookup(call_unit_harness):
    def slow(v):
        time.sleep(0.05)
        return v

    unit = call_unit_harness(lineage={"x": "h"}, user_ns={"x": 1, "slow": slow})
    site = CallSite(source="slow(x)", free_names=frozenset({"slow", "x"}), occurrence_index=0)
    unit.wrap(slow, site)(1)

    events = unit.drain()
    assert len(events) == 1
    assert events[0]["intercepted"] is True
    assert events[0]["call_source"] == "slow(x)"
    assert unit.drain() == [], "drain must clear"


def test_a_cache_hit_is_also_marked_intercepted(call_unit_harness):
    """The flag must not be an artefact of the miss path only."""
    def slow(v):
        # `_storable` refuses to cache a call whose result IS one of its
        # arguments (identity), so this must return something new rather
        # than `v` itself or the entry would never be stored at all.
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)  # above the cost-model floor
        return v + 1

    unit = call_unit_harness(lineage={"x": "h"}, user_ns={"x": 1, "slow": slow})
    site = CallSite(source="slow(x)", free_names=frozenset({"slow", "x"}), occurrence_index=0)
    wrapped = unit.wrap(slow, site)
    wrapped(1)
    unit.drain()  # discard the miss event
    wrapped(1)     # now a hit

    events = unit.drain()
    assert len(events) == 1
    assert events[0]["cache_hit"] is True
    assert events[0]["intercepted"] is True


def test_call_unit_events_reach_the_text_badge_as_intercepted(call_unit_harness):
    """End-to-end through the exact renderer the processor feeds: a call-unit
    event, unmodified apart from the ``intercepted`` flag CallUnit itself set,
    must render with the ``[intercepted]`` marker -- proving the badge no longer
    needs (or gets) a separate reconciliation pass to show this.
    """
    def slow(v):
        time.sleep(0.05)
        return v

    unit = call_unit_harness(lineage={"x": "h"}, user_ns={"x": 1, "slow": slow})
    site = CallSite(source="slow(x)", free_names=frozenset({"slow", "x"}), occurrence_index=0)
    unit.wrap(slow, site)(1)
    events = unit.drain()

    metrics = [{
        "status": "COMPUTED",
        "code": "out = slow(x)",
        "total_time": 0.4,
        "evaluated_vars": ["out"],
        "decorator_calls": events,
        "is_upstream": False,
    }]
    text = render_text(build_interactive_badge(metrics))
    assert "[intercepted]" in text, (
        f"a call-unit event was not rendered as intercepted:\n{text}"
    )

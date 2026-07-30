"""An intercepted call must be legible as such on the badge (CAS-243).

Interception is on by DEFAULT (task 10) — no directive needed to trigger it,
``# @cash:no-cache-calls`` is the opt-out. Intercepted calls land in the same
"N calls, M cached" region as hand-decorated ones, which is the right place —
it is the same cache. But undifferentiated it misleads in both directions:

- a user who decorated nothing sees a ``@cash.cache`` section appear and cannot
  tell where it came from;
- a user relying on interception (which is unconditional now) has no way to
  confirm it actually engaged for a given call.

So the group carries the distinction and the renderers show it, tagged
``@intercepted`` -- deliberately NOT named after a directive, since (a)
nothing needs to be written to trigger it and (b) a tag spelled
``cache-calls`` would substring-collide with ``no-cache-calls`` in any
grep/assertion over badge text.
"""
from __future__ import annotations

import types

import pytest

import cash
from cash.notebook.badge_renderer.view_builder import build_interactive_badge
from cash.notebook.badge_renderer.renderers.text import render_text
from cash.notebook.call_interception import CallCache, CallSite
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


# ---------------------------------------------------------------- CallCache

@pytest.fixture
def call_cache(tmp_path):
    return CallCache(cash.Cash(cache_dir=str(tmp_path / "cc")))


def _site(source="compute(x)", names=("compute", "x")):
    return CallSite(
        source=source, free_names=frozenset(names), occurrence_index=0,
        computed_arg_positions=(0,),
    )


def test_wrapped_calls_are_recorded_as_intercepted(call_cache):
    """Every event a genuinely-wrapped call produces must carry the flag the
    badge reads -- set at the source now, not reconciled by name afterwards.
    """
    def compute(x):
        import time
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)  # above the cost-model floor
        return x + 1

    call_cache.set_sites([_site()])
    wrapped = call_cache.resolve(compute)
    wrapped(5)

    events = call_cache.drain_call_log()
    assert events, "the wrapped call produced no drained event"
    assert all(e["intercepted"] is True for e in events), events


def test_passed_through_callables_produce_no_event(call_cache):
    """Only what was actually wrapped logs anything, or the badge over-claims."""
    call_cache.set_sites([_site(source="len(a)", names=("len", "a"))])
    call_cache.resolve(len)
    call_cache.resolve(None)
    assert call_cache.drain_call_log() == []


def test_already_decorated_functions_produce_no_call_cache_event(call_cache, tmp_path):
    """A hand-decorated call must keep reading as hand-decorated: resolving it
    through ``call_cache`` must not wrap it, so nothing lands in the
    call-unit log this ``call_cache`` drains -- calling it is accounted for
    by the decorator's own log instead (a separate mechanism entirely).
    """
    other = cash.Cash(cache_dir=str(tmp_path / "other"))

    @other.cache
    def compute(x):
        return x + 1

    call_cache.set_sites([_site()])
    resolved = call_cache.resolve(compute)
    resolved(5)
    assert call_cache.drain_call_log() == []


# ------------------------------------------------------------- rendering

def _metrics(intercepted: bool):
    return [{
        "status": "COMPUTED",
        "code": "out.append(compute(x))",
        "total_time": 0.4,
        "evaluated_vars": ["out"],
        "decorator_calls": [
            {"func_name": "__main__.compute", "cache_hit": True,
             "execution_time": 0.001, "intercepted": intercepted},
            {"func_name": "__main__.compute", "cache_hit": False,
             "execution_time": 0.2, "intercepted": intercepted},
        ],
        "is_upstream": False,
    }]


def test_text_badge_marks_an_intercepted_group():
    text = render_text(build_interactive_badge(_metrics(intercepted=True)))
    assert "compute()" in text, text
    assert "[intercepted]" in text, (
        f"an intercepted group must be labelled as such:\n{text}"
    )


def test_text_badge_leaves_a_decorated_group_unmarked():
    """Positive control: the marker must not appear for ordinary decorated calls."""
    text = render_text(build_interactive_badge(_metrics(intercepted=False)))
    assert "compute()" in text, text
    assert "[intercepted]" not in text, (
        f"a hand-decorated group was mislabelled as intercepted:\n{text}"
    )


def _html(intercepted: bool, n_calls: int = 2) -> str:
    from cash.notebook.badge_renderer.renderers.html import render_html
    metrics = _metrics(intercepted)
    call = metrics[0]["decorator_calls"][0]
    metrics[0]["decorator_calls"] = [dict(call) for _ in range(n_calls)]
    return render_html(build_interactive_badge(metrics))


def test_html_badge_marks_an_intercepted_group():
    """A notebook shows HTML badges by default, so the marker must reach there too."""
    assert "@intercepted" in _html(intercepted=True), (
        "the HTML badge does not distinguish an intercepted call"
    )


def test_html_badge_marks_an_intercepted_group_when_condensed():
    """The >3-call path renders the group directly and must mark it too.

    The two HTML paths are separate code: below the condense threshold each
    call renders its own row and never sees the group, so marking only the
    condensed branch would leave short loops unlabelled.
    """
    assert "@intercepted" in _html(intercepted=True, n_calls=5)


def test_html_badge_leaves_decorated_groups_unmarked():
    """Positive control, both paths."""
    assert "@intercepted" not in _html(intercepted=False, n_calls=2)
    assert "@intercepted" not in _html(intercepted=False, n_calls=5)


def test_absent_flag_reads_as_decorated():
    """Metrics from before this field existed must not start claiming interception."""
    metrics = _metrics(intercepted=False)
    for call in metrics[0]["decorator_calls"]:
        del call["intercepted"]
    text = render_text(build_interactive_badge(metrics))
    assert "[intercepted]" not in text, text

"""Waste attribution must not manufacture waste out of threshold noise.

cash declines to cache anything under a ~10ms floor. Near that floor the
cost model's own decision flips on timing noise, so a statement can be
cached during the control run and refused during the edited one. The
comparison then reads "restored, then recomputed" and reports waste --
with no invalidation involved anywhere.

That is how the CI floor test went red on macos-3.10 and nowhere else:
``base = np.arange(2_000_000)`` costs 2.2ms on a fast box, so it is never
cached and never counted, and straddles the floor on a slow one.

The guard is a floor on the recompute cost. The risk of a guard like this
is that it quietly disables the check it protects, so the control below
matters more than the positive case.
"""
from __future__ import annotations

from benchmarks._edit_scenarios import (
    MIN_WASTE_SECONDS,
    EditScenario,
    attribute_waste,
)
from benchmarks._overhead_results import StatementMetric


def _m(code, status, seconds):
    return StatementMetric(code=code, execution_time=seconds,
                           total_time=seconds, status=status)


def _attribute(recompute_seconds):
    """One restorable statement downstream of the edit, recomputed after it."""
    scenario = EditScenario(kind="null-assign", site=0, label="null-assign@cell0")
    control = {0: [_m("edited = 1", "COMPUTED", 0.0)],
               1: [_m("heavy = work()", "RESTORED", 0.5)]}
    edited = {0: [_m("edited = 1", "COMPUTED", 0.0)],
              1: [_m("heavy = work()", "COMPUTED", recompute_seconds)]}
    return attribute_waste(scenario, control, edited)


def test_real_waste_is_still_counted():
    """The control arm. A guard that silences everything passes vacuously."""
    result = _attribute(recompute_seconds=0.5)
    assert result.wasted_count == 1, "genuine waste was silenced by the guard"
    assert result.wasted[0].execution_seconds == 0.5


def test_a_recompute_below_the_margin_is_not_called_waste():
    result = _attribute(recompute_seconds=MIN_WASTE_SECONDS / 4)
    assert result.wasted_count == 0, (
        f"counted a {MIN_WASTE_SECONDS / 4:.3f}s recompute as waste; near the "
        f"caching floor the cost model's own decision is timing noise, so this "
        f"reports over-invalidation that did not happen"
    )


def test_the_margin_sits_above_the_caching_floor():
    """Not an arbitrary number: it has to clear the ~10ms floor with room.

    At exactly the floor the flip-flop this guards against is most likely,
    so the margin is 2x rather than 1x.
    """
    assert MIN_WASTE_SECONDS >= 0.020


def test_the_boundary_is_inclusive_upward():
    """Just above the margin counts; just below does not."""
    assert _attribute(MIN_WASTE_SECONDS * 1.01).wasted_count == 1
    assert _attribute(MIN_WASTE_SECONDS * 0.99).wasted_count == 0

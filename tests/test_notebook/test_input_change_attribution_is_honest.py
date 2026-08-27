"""``input changed: x`` must not be reported when x did not change.

``_attribute_input_change`` names the input whose change forced a
statement to re-run. It works by comparing ``executed_input_lineages``
(what the statement last RAN with) against the current lineage -- and
that record is keyed by **output variable name only**.

So every statement writing the same variable shares one slot. In a
filter chain::

    df = pd.read_csv(...)      # writes df
    df = df[df.a > 0]          # writes df, clobbering the slot
    df = df[df.b < 9]          # writes df, clobbering it again

each statement reads the record left by whichever ran last, not its own.
Measured on 01_nyc_taxi_analysis: 21 attributions in one run, **17** of
them naming an input that is also the statement's own output, on a run
where the cache keys and df's whole lineage sequence were byte-identical
to the previous one. Nothing had changed; the badge said something had.

A wrong reason is worse than no reason. The row already rendered
EXECUTED before this feature existed, and a user chasing "input changed:
df" has been sent to look at a variable that is not the problem.

Tested against the attributor directly rather than through a kernel: the
comparison is the bug, and it needs only a tracking state to exercise.
"""
from __future__ import annotations

from types import SimpleNamespace

from cash.notebook.statement.processor import StatementProcessor


def _attribute(*, variable_lineage, executed_input_lineages, inputs, outputs):
    """Run the attributor over a minimal tracking state, return miss_reason."""
    stub = SimpleNamespace(_tracking_state=SimpleNamespace(
        variable_lineage=dict(variable_lineage),
        executed_input_lineages=dict(executed_input_lineages),
    ))
    metrics: dict = {}
    StatementProcessor._attribute_input_change(stub, metrics, inputs, outputs)
    return metrics.get("miss_reason")


def test_a_self_referential_statement_does_not_blame_its_own_output():
    """``df = df[mask]`` must not report ``input changed: df``.

    The statement both reads and writes ``df``, so the slot it is compared
    against was written by whichever statement wrote ``df`` last -- often a
    different one in the same chain. The comparison cannot distinguish "my
    input changed" from "someone else advanced this variable", so it must
    not assert the former.
    """
    reason = _attribute(
        variable_lineage={"df": "lineage-2"},
        executed_input_lineages={"df": {"df": "lineage-1"}},
        inputs={"df"},
        outputs={"df"},
    )
    assert reason is None, (
        f"attributed {reason!r} for a statement whose 'changed' input is its "
        f"own output; the record it compared against belongs to whichever "
        f"statement wrote that variable last, not necessarily this one"
    )


def test_a_genuine_upstream_change_is_still_named():
    """The control arm: fixing the lie must not silence the truth.

    ``agg = df.groupby(...)`` writes ``agg`` and only reads ``df``, so the
    record under ``agg`` was written by this statement and the comparison
    is sound. This is the case the feature exists for.
    """
    reason = _attribute(
        variable_lineage={"df": "lineage-2"},
        executed_input_lineages={"agg": {"df": "lineage-1"}},
        inputs={"df"},
        outputs={"agg"},
    )
    assert reason == "input changed: df", reason


def test_an_unchanged_input_is_never_named():
    """Baseline: matching lineage means the statement did not re-run for it."""
    reason = _attribute(
        variable_lineage={"df": "lineage-1"},
        executed_input_lineages={"agg": {"df": "lineage-1"}},
        inputs={"df"},
        outputs={"agg"},
    )
    assert reason is None, reason


def test_a_partly_self_referential_statement_still_names_the_other_input():
    """``df = df.join(other)`` may honestly blame ``other``.

    Only the self-referential name is untrustworthy. Suppressing the whole
    attribution would throw away a reason that is still sound.
    """
    reason = _attribute(
        variable_lineage={"df": "lineage-2", "other": "lineage-9"},
        executed_input_lineages={"df": {"df": "lineage-1", "other": "lineage-8"}},
        inputs={"df", "other"},
        outputs={"df"},
    )
    assert reason == "input changed: other", reason

"""An output the identity walk gives up on (more objects than it looks
through) is refused before the alias check walks the same objects again; an
output it passes is still checked for live aliases."""

from __future__ import annotations

import types

import pytest

from cash.notebook.statement import identity_holders, output_refusals
from cash.notebook.statement.identity_holders import forget_flat_reads
from cash.notebook.statement.output_refusals import unrestorable_output_reason

np = pytest.importorskip("numpy")


def _alias_checks(monkeypatch) -> list[str]:
    asked: list[str] = []
    check = output_refusals._alias_refusal

    def counted(name, *args, **kwargs):
        asked.append(name)
        return check(name, *args, **kwargs)

    monkeypatch.setattr(output_refusals, "_alias_refusal", counted)
    return asked


def test_a_value_past_the_identity_budget_is_not_walked_for_aliases(monkeypatch):
    monkeypatch.setattr(identity_holders, "_MAX_SEEN", 100)
    asked = _alias_checks(monkeypatch)
    # Not plain data, which is read whatever its size (below).
    big = [types.SimpleNamespace(i=i) for i in range(500)]
    ns = {"big": big}
    forget_flat_reads()
    reason = unrestorable_output_reason({"big"}, {"big": big}, ns)
    assert reason is not None and "more objects than cash looks through" in reason
    assert asked == []


def test_plain_data_past_the_identity_budget_is_still_walked_for_aliases(monkeypatch):
    """Lists of ints are read whole, past the count, and pass the identity
    walk: the alias check still looks."""
    monkeypatch.setattr(identity_holders, "_MAX_SEEN", 100)
    asked = _alias_checks(monkeypatch)
    big = [[i] for i in range(500)]
    forget_flat_reads()
    reason = unrestorable_output_reason({"big"}, {"big": big}, {"big": big})
    assert reason is None or "more objects than cash looks through" not in reason
    assert asked == ["big"]


def test_a_view_the_identity_walk_passes_is_still_refused_as_an_alias(monkeypatch):
    asked = _alias_checks(monkeypatch)
    base = np.linspace(0.0, 1.0, 10)
    view = base[2:5]
    ns = {"base": base, "view": view}
    forget_flat_reads()
    reason = unrestorable_output_reason({"view"}, {"view": view}, ns)
    assert reason is not None and reason.startswith("Live-alias object 'view'")
    assert asked == ["view"]

"""Named mutations: deliberate breakages used to measure what the suite catches.

A green suite proves nothing on its own. It proves something once you can
break a subsystem and watch the suite go red -- and the interesting cases are
the ones where it *doesn't*, because that is a subsystem nobody is testing.

Each entry breaks one mechanism as bluntly as possible. Subtlety is wrong here:
a mutation that only breaks an edge case tells you nothing about whether the
mainline is covered.

Adding one: give it a name, a target module, and an ``apply`` that patches the
module in place. ``apply`` must route every call through ``record`` so the
harness can tell "the suite tolerated a broken engine" apart from "the mutation
never actually ran" -- those look identical from outside and mean opposite
things.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Mutation:
    name: str
    target: str          #: module that must be loaded before ``apply`` runs
    probe: str           #: attribute whose presence means the module is ready
    replaces: tuple      #: ("Class.method", ...) this overwrites -- each MUST exist
    description: str
    apply: Callable      #: (module, record) -> None


def _upstream_dead(mod, record) -> None:
    """The upstream re-execution decision schedules nothing and restores nothing.

    Kills cell-to-cell invalidation outright: after this, editing an upstream
    cell can no longer cause a downstream statement to re-run.
    """
    def dead(self, simulation_trace, broken_vars, *a, **kw):
        record()
        return [], [], 0.0

    mod.MismatchClassifier._backward_scan_pass = dead


def _restore_always_fails(mod, record) -> None:
    """Virtual restore never succeeds, so everything must re-execute.

    The inverse of ``upstream-dead``: correctness is preserved and only speed
    is destroyed. A suite that catches this but not ``upstream-dead`` is
    testing performance, not invalidation.
    """
    original = mod.VirtualLineage._try_virtual_restore

    def never(self, *a, **kw):
        record()
        return set(), 0.0, 0.0

    mod.VirtualLineage._try_virtual_restore = never
    del original


def _file_deps_always_fresh(mod, record) -> None:
    """A changed file dependency is never noticed, so a stale value is served.

    The staleness bug a caching tool most needs to be caught doing. Returning
    ``cached_data`` unchanged is this method's "still fresh" answer; returning
    None is how it signals invalidation.
    """
    def never_invalidates(self, metadata, cached_data, *a, **kw):
        record()
        return cached_data

    mod.CacheFreshnessChecker._invalidate_if_direct_file_changed = never_invalidates


CATALOGUE: dict[str, Mutation] = {
    m.name: m for m in (
        Mutation(
            name="upstream-dead",
            target="cash.notebook.upstream.mismatch_classifier",
            probe="MismatchClassifier",
            replaces=("MismatchClassifier._backward_scan_pass",),
            description="upstream re-execution decision schedules nothing",
            apply=_upstream_dead,
        ),
        Mutation(
            name="restore-dead",
            target="cash.notebook.upstream.virtual_lineage",
            probe="VirtualLineage",
            replaces=("VirtualLineage._try_virtual_restore",),
            description="virtual restore never succeeds; everything re-executes",
            apply=_restore_always_fails,
        ),
        Mutation(
            name="file-deps-blind",
            target="cash.notebook.statement.freshness",
            probe="CacheFreshnessChecker",
            replaces=("CacheFreshnessChecker._invalidate_if_direct_file_changed",),
            description="file-dependency changes are never noticed",
            apply=_file_deps_always_fresh,
        ),
    )
}

"""What a statement was seen to change besides the names its text binds.

The runtime learns part of what a statement changes from the objects it
meets: ``f(4)`` changes ``lst`` when ``f`` holds ``lst.append``, ``inc()``
changes the closure ``inc`` holds, ``getcontext().prec = 6`` changes a
setting of the process. The upstream simulation predicts the same facts for
the statements above a cell without running them, and after a kernel restart
it has no object to ask. :class:`EffectsRecord` is the one record the run
leaves for it: kept for the session (``TrackingState.effects_records``) and
on disk under the statement's source hash (``statement_effects_key``).

A record only ever adds to what the simulation reads from the statement's
text: a variable a run was seen to change is a variable the statement may
change, so it runs again where a reader below needs the variable. It never
says a statement changes nothing.

The names a run changed are kept under the lineages its inputs had
(:func:`inputs_digest`), a few runs deep: asked with the inputs the
simulation gives the statement, the record answers with that run's names,
else with those of every run it holds.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["RUNS_KEPT", "EffectsRecord", "inputs_digest"]

#: How many runs, each with other input lineages, a record keeps the changed
#: names of; the oldest goes first.
RUNS_KEPT = 16


def inputs_digest(lineages: Mapping[str, str]) -> str:
    """One short digest of the lineages a statement's inputs have: the same
    for the run and for a simulation that gives its inputs those lineages."""
    text = "\n".join(f"{name}={lineage}" for name, lineage in sorted(lineages.items()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class EffectsRecord:
    """What the runs of one statement were seen to change."""

    #: ``(inputs digest, names)`` of each run kept, oldest first: the
    #: variables the run changed that the statement's text does not bind.
    runs: tuple[tuple[str, frozenset[str]], ...] = ()
    #: What of the process its last run changed (``callee_reach.ENVIRON``,
    #: ``CWD``, ``REGISTRY``).
    process: frozenset[str] = frozenset()

    def changed(self, at: str | None = None) -> frozenset[str]:
        """The names the run with the inputs *at* changed; those of every
        run kept when no run had these inputs."""
        every: set[str] = set()
        for digest, names in self.runs:
            if digest == at:
                return names
            every |= names
        return frozenset(every)

    def with_run(self, at: str, names: Iterable[str]) -> EffectsRecord:
        """The record after a run with the inputs *at* changed *names*."""
        names = frozenset(names)
        runs = [run for run in self.runs if run[0] != at]
        runs.append((at, names))
        return EffectsRecord(tuple(runs[-RUNS_KEPT:]), self.process)

    def with_process(self, kinds: Iterable[str]) -> EffectsRecord:
        """The record after a run changed *kinds* of the process."""
        return EffectsRecord(self.runs, frozenset(kinds))

    def to_dict(self) -> dict[str, Any]:
        """The record as the backend keeps it."""
        return {
            "effects": True,
            "runs": [[digest, sorted(names)] for digest, names in self.runs],
            "process": sorted(self.process),
            "ttl": None,
        }

    @classmethod
    def from_dict(cls, record: Any) -> EffectsRecord | None:
        """The record *record* holds, or None when it is not one."""
        if not isinstance(record, Mapping) or not record.get("effects"):
            return None
        try:
            runs = tuple(
                (str(digest), frozenset(str(name) for name in names)) for digest, names in record.get("runs") or ()
            )
            process = frozenset(str(kind) for kind in record.get("process") or ())
        except (TypeError, ValueError):
            return None
        return cls(runs, process)

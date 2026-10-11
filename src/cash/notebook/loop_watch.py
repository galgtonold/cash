"""What a loop changes through code whose effect its text cannot say.

``heapq.heappush(h, x)``, ``np.copyto(a, b)``, ``rng.shuffle(xs)``,
``if heapq.heappop(h) > t:`` and ``x = dq.popleft()`` change what they are
handed, and nothing in the loop's text tells a library function that changes
its argument from one that reads it. The candidates
(``mutation_effects.loop_watch_candidates``) are fingerprinted before and
after the loop runs, the first time this loop's source runs in a session,
and the names that changed are recorded under the loop's source, as a
statement's verdict is (``mutation_verdicts``, persisted for a later
kernel). The loop's lineage update, a restored loop and the simulation all
apply that record, so the three move the same names.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..analysis.mutation_effects import loop_watch_candidates
from ..mutation_fingerprint import mutation_fingerprint
from .cache_key import statement_source_hash
from .consumables import is_stream

__all__ = ["LoopWatch", "loop_watch_hash", "watch_candidates"]


def loop_watch_hash(code: str) -> str:
    """The name the verdict of the loop *code* is recorded under: apart from
    the verdict a statement of the same text records for its own calls."""
    return hashlib.sha256(f"loop-watch:{statement_source_hash(code)}".encode()).hexdigest()


def watch_candidates(
    node: ast.AST,
    namespace: Mapping[str, Any],
    is_module: Callable[[str], bool],
    resolve_source: Callable[[str], str | None] | None,
) -> frozenset[str]:
    """``loop_watch_candidates`` without the streams: an iterator, a queue or
    a file is followed by its draws (``drawn_stream_inputs``), and a
    fingerprint of one could only say it changed."""
    names = loop_watch_candidates(node, namespace, is_module, resolve_source)
    return frozenset(name for name in names if not is_stream(namespace.get(name)))


@dataclass
class LoopWatch:
    """One loop's candidates, fingerprinted before it ran.

    *verdict* is what an earlier run of this loop source recorded; when it is
    known nothing is fingerprinted. *before* holds each candidate's
    fingerprint otherwise (None: it cannot be observed, so it counts as
    changed).
    """

    key: str
    candidates: frozenset[str]
    verdict: frozenset[str] | None
    before: dict[str, str | None] = field(default_factory=dict)
    #: The names it changed, once :meth:`finish` saw the loop run.
    seen: frozenset[str] | None = None

    @classmethod
    def start(
        cls, code: str, candidates: frozenset[str], verdict: set[str] | None, namespace: Mapping[str, Any]
    ) -> LoopWatch:
        known = frozenset(verdict) if verdict is not None else None
        watch = cls(loop_watch_hash(code), candidates, known)
        if known is None:
            watch.before = {name: _fingerprint(namespace.get(name)) for name in candidates}
        return watch

    @property
    def learning(self) -> bool:
        """Whether this run decides the verdict: the loop must then really
        run, not be restored, or what it changes is never seen."""
        return self.verdict is None and bool(self.candidates)

    def finish(self, namespace: Mapping[str, Any]) -> frozenset[str]:
        """The candidates the loop changed: seen now when it was being
        learned, else the recorded verdict."""
        if self.seen is not None:
            return self.seen
        if self.verdict is not None:
            self.seen = self.verdict
            return self.seen
        changed = set()
        for name, before in self.before.items():
            if before is None or name not in namespace or _fingerprint(namespace[name]) != before:
                changed.add(name)
        self.seen = frozenset(changed)
        return self.seen

    def assumed(self) -> frozenset[str]:
        """What a loop restored without running changed: the recorded
        verdict, else every candidate (a restore cannot watch it)."""
        if self.verdict is not None:
            return self.verdict
        return self.candidates


def _fingerprint(value: Any) -> str | None:
    try:
        return mutation_fingerprint(value)
    except Exception:  # noqa: BLE001 - unobservable: counted as changed
        return None

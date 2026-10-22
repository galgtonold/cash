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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from cash._user_attrs import attribute_or

from ..analysis.mutation_effects import loop_watch_candidates, statement_watch_candidates
from ..mutation_fingerprint import mutation_fingerprint
from .cache_key import statement_source_hash
from .consumables import is_stream

__all__ = ["LoopWatch", "branch_changes", "loop_watch_hash", "watch_candidates"]


def loop_watch_hash(code: str) -> str:
    """The name the verdict of the loop *code* is recorded under: apart from
    the verdict a statement of the same text records for its own calls."""
    return hashlib.sha256(f"loop-watch:{statement_source_hash(code)}".encode()).hexdigest()


def branch_changes(
    node: ast.AST,
    verdict: Callable[[str], Iterable[str] | None],
    namespace: Mapping[str, Any],
    is_module: Callable[[str], bool],
    resolve_source: Callable[[str], str | None] | None,
) -> set[str]:
    """What an ``if`` or ``try`` *node* changes through calls its text cannot
    decide: per statement in its branches, which runs one at a time and
    learns what it changes when it really runs, its recorded *verdict*; per
    loop or ``with`` in them, watched as a unit, that one's. One that has
    not run yet (a branch never taken) counts every name it hands to such
    code, as a loop not yet watched does.

    ``if use_heap: heapq.heappush(h, k)`` changes ``h``. The runtime's
    lineage update for the ``if`` and the simulation of it call this with
    their own records and namespace, so they move the same names. Empty for
    any other node.
    """
    if not isinstance(node, (ast.If, ast.Try)):
        return set()
    changed: set[str] = set()

    def walk(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(stmt, (ast.If, ast.Try)):
                walk(_branches_of(stmt))
                continue
            if isinstance(stmt, (ast.For, ast.While, ast.With)):
                seen = verdict(loop_watch_hash(ast.unparse(stmt)))
                if seen is None:
                    seen = watch_candidates(stmt, namespace, is_module, resolve_source)
                changed.update(seen)
                walk(_branches_of(stmt))
                continue
            seen = verdict(statement_source_hash(ast.unparse(stmt)))
            if seen is None:
                tree = ast.Module(body=[stmt], type_ignores=[])
                seen = statement_watch_candidates(tree, namespace, is_module, resolve_source)
            changed.update(seen)

    walk(_branches_of(node))
    return changed


def _branches_of(node: ast.AST) -> list[ast.stmt]:
    body: list[ast.stmt] = [*attribute_or(node, "body", ()), *attribute_or(node, "orelse", ())]
    for handler in attribute_or(node, "handlers", ()):
        body.extend(handler.body)
    body.extend(attribute_or(node, "finalbody", ()))
    return body


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

    *verdict* is what earlier runs of this loop source were seen changing.
    It only grows: a run that changed nothing (no items yet, a branch not
    taken) says nothing about a run on other inputs. So a run on inputs
    not observed before (*rewatch*) fingerprints the candidates the verdict
    does not hold yet, and a change seen joins the verdict. *before* holds
    each fingerprinted candidate's fingerprint (None: it cannot be
    observed, so it counts as changed).
    """

    key: str
    candidates: frozenset[str]
    verdict: frozenset[str] | None
    before: dict[str, str | None] = field(default_factory=dict)
    #: The names it changed, once :meth:`finish` saw the loop run.
    seen: frozenset[str] | None = None

    @classmethod
    def start(
        cls,
        code: str,
        candidates: frozenset[str],
        verdict: set[str] | None,
        namespace: Mapping[str, Any],
        rewatch: bool = False,
    ) -> LoopWatch:
        known = frozenset(verdict) if verdict is not None else None
        watch = cls(loop_watch_hash(code), candidates, known)
        if known is None:
            watch.before = {name: _fingerprint(namespace.get(name)) for name in candidates}
        elif rewatch:
            watch.before = {name: _fingerprint(namespace.get(name)) for name in candidates - known}
        return watch

    @property
    def learning(self) -> bool:
        """Whether this run decides the verdict: the loop must then really
        run, not be restored, or what it changes is never seen."""
        return self.verdict is None and bool(self.candidates)

    @property
    def watching(self) -> bool:
        """Whether this run fingerprinted something to compare after it."""
        return bool(self.before)

    def finish(self, namespace: Mapping[str, Any]) -> frozenset[str]:
        """The candidates the loop changed: the recorded verdict, with what
        the fingerprints taken before it ran show it changed now."""
        if self.seen is not None:
            return self.seen
        changed = set(self.verdict or ())
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

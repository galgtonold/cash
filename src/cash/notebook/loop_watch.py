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
import copy
import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from cash._user_attrs import attribute_or

from ..analysis.ast_util import walk_tree
from ..analysis.mutation_effects import loop_watch_candidates, statement_watch_candidates
from ..mutation_fingerprint import mutation_fingerprint
from .cache_key import statement_source_hash
from .consumables import is_stream

__all__ = [
    "LoopWatch",
    "branch_changes",
    "loop_handed_hash",
    "loop_watch_hash",
    "surely_handed",
    "watch_candidates",
]


def loop_watch_hash(code: str) -> str:
    """The name the verdict of the loop *code* is recorded under: apart from
    the verdict a statement of the same text records for its own calls."""
    return hashlib.sha256(f"loop-watch:{statement_source_hash(code)}".encode()).hexdigest()


def loop_handed_hash(code: str) -> str:
    """The name the record of what the loop *code* was seen handing is kept
    under (:meth:`LoopWatch.finish`): the candidates a call of its body was
    handed, in a run where that call ran, fingerprinted around it."""
    return hashlib.sha256(f"loop-handed:{statement_source_hash(code)}".encode()).hexdigest()


#: What evaluates only some of an expression, or later, or not at all: a
#: call under one may not have run when the statement did.
_PARTLY_EVALUATED = (
    ast.Lambda,
    ast.IfExp,
    ast.BoolOp,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)

#: What ends a pass early: a statement after one may not run on a pass.
_JUMPS = (ast.Break, ast.Continue, ast.Return, ast.Raise)


def surely_handed(
    node: ast.AST,
    namespace: Mapping[str, Any],
    is_module: Callable[[str], bool],
    resolve_source: Callable[[str], str | None] | None,
) -> tuple[dict[int, frozenset[str]], int]:
    """Per statement of the ``for`` loop *node*'s body, by its index, the
    candidates (`watch_candidates`) its calls are handed whenever it runs,
    which no other part of the loop hands; and how many of the body's first
    statements run on every pass that starts (none of them before can end
    the pass).

    A statement counts only when every call in it runs whenever it does: a
    plain statement with no ``and``/``or``, conditional expression,
    comprehension or ``lambda`` in it, calling no function of the user's
    (whose body may hand on only some of what it is handed). A name another
    statement, the header or the ``else`` hands too is no statement's:
    ``model.fit(X)`` then ``if c: scale_inplace(X)`` leaves ``X`` out.

    ``while`` loops and ``with`` blocks have none: nothing tells whether
    their body ran."""
    if not isinstance(node, ast.For):
        return {}, 0
    whole = [i for i, stmt in enumerate(node.body) if _runs_whole(stmt, namespace, resolve_source)]
    if not whole:
        return {}, 0
    quiet = [_without_calls(stmt) for stmt in node.body]

    def handed_by(body: list[ast.stmt], orelse: list[ast.stmt]) -> frozenset[str]:
        loop = ast.For(target=node.target, iter=node.iter, body=body, orelse=orelse, type_comment=None)
        ast.fix_missing_locations(loop)
        return watch_candidates(loop, namespace, is_module, resolve_source)

    rest = handed_by([quiet[i] if i in whole else stmt for i, stmt in enumerate(node.body)], list(node.orelse))
    handed: dict[int, frozenset[str]] = {}
    for i in whole:
        names = handed_by([stmt if j == i else quiet[j] for j, stmt in enumerate(node.body)], []) - rest
        if names:
            handed[i] = names
    reach = 0
    for stmt in node.body:
        reach += 1
        if any(isinstance(n, _JUMPS) for n in _own_nodes(stmt)):
            break
    return handed, reach


def _runs_whole(stmt: ast.stmt, namespace: Mapping[str, Any], resolve_source) -> bool:
    """Whether every call in *stmt* runs whenever *stmt* does (`surely_handed`)."""
    if not isinstance(stmt, (ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign)):
        return False
    for sub in walk_tree(stmt):
        if isinstance(sub, _PARTLY_EVALUATED):
            return False
        if isinstance(sub, ast.Call):
            name = _dotted_name(sub.func)
            if name is None:
                continue
            if resolve_source is not None:
                try:
                    if resolve_source(name) is not None:
                        return False
                except Exception:  # noqa: BLE001 - unreadable: counted as the user's
                    return False
    return True


def _dotted_name(func: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if not isinstance(func, ast.Name):
        return None
    parts.append(func.id)
    return ".".join(reversed(parts))


def _own_nodes(stmt: ast.stmt) -> list[ast.AST]:
    """The nodes of *stmt* outside the functions and classes it defines."""
    found: list[ast.AST] = []
    pending: list[ast.AST] = [stmt]
    while pending:
        current = pending.pop()
        found.append(current)
        for child in ast.iter_child_nodes(current):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                pending.append(child)
    return found


class _NoCalls(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        return ast.copy_location(ast.Constant(value=None), node)


def _without_calls(stmt: ast.stmt) -> ast.stmt:
    """*stmt* with every call taken out: what it binds stays, for the
    context of the statements around it, but it hands nothing over."""
    return ast.fix_missing_locations(_NoCalls().visit(copy.deepcopy(stmt)))


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
    #: Per statement of a ``for`` body, what its calls are surely handed
    #: when it runs, and how many first statements run on every pass
    #: (`surely_handed`).
    handed: dict[int, frozenset[str]] = field(default_factory=dict)
    reach: int = 0
    #: What the loop's targets were bound to before it ran (`_ABSENT`: not
    #: bound): one bound to another object after it ran says a pass began.
    targets_before: dict[str, Any] = field(default_factory=dict)
    #: Run pass by pass: only the statements seen running (:meth:`ran`) say
    #: what was handed, not the targets the passes bound.
    passes: bool = False
    ran_statements: set[int] = field(default_factory=set)
    #: What :meth:`finish` saw handed and fingerprinted around the run.
    seen_handed: frozenset[str] = frozenset()

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

    def follow_passes(
        self,
        node: ast.AST,
        handed: dict[int, frozenset[str]],
        reach: int,
        namespace: Mapping[str, Any],
    ) -> None:
        """Take what the statements of the ``for`` loop *node* surely hand
        (`surely_handed`), and what its targets hold before it runs."""
        self.handed = {i: names & frozenset(self.before) for i, names in handed.items()}
        self.handed = {i: names for i, names in self.handed.items() if names}
        self.reach = reach
        if not self.handed or not isinstance(node, ast.For):
            return
        # A target the header binds itself (``for x in (x := xs)``) is bound
        # whether or not a pass began.
        header = {n.target.id for n in walk_tree(node.iter) if isinstance(n, ast.NamedExpr)}
        targets = {n.id for n in walk_tree(node.target) if isinstance(n, ast.Name)}
        if targets & header:
            return
        self.targets_before = {name: namespace.get(name, _ABSENT) for name in targets}

    def ran(self, index: int) -> None:
        """Statement *index* of the body ran on a pass (not served)."""
        self.ran_statements.add(index)

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
        self.seen_handed = frozenset().union(*(self.handed[i] for i in self._statements_run(namespace)))
        self.targets_before = {}
        return self.seen

    def _statements_run(self, namespace: Mapping[str, Any]) -> set[int]:
        """The statements of the body known to have run: those seen running
        pass by pass, else, run as one unit, the first ones every pass runs
        once a target holds another object than before."""
        if self.passes:
            return self.ran_statements & set(self.handed)
        if not self.targets_before:
            return set()
        began = any(namespace.get(name, _ABSENT) is not before for name, before in self.targets_before.items())
        return {i for i in self.handed if i < self.reach} if began else set()

    def assumed(self) -> frozenset[str]:
        """What a loop restored without running changed: the recorded
        verdict, else every candidate (a restore cannot watch it)."""
        if self.verdict is not None:
            return self.verdict
        return self.candidates


_ABSENT = object()


def _fingerprint(value: Any) -> str | None:
    try:
        return mutation_fingerprint(value)
    except Exception:  # noqa: BLE001 - unobservable: counted as changed
        return None

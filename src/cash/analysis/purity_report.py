"""What the purity analysis of a ``@cash.cache`` function reports.

The kinds of finding (``ISSUE_*``), one finding (:class:`PurityIssue`) and the
whole report for a function and the helpers it reaches (:class:`PurityReport`),
which the decorator turns into warnings, errors and cache-key parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..effects import EffectKind

ISSUE_IMPURE_CALL = "impure_call"
ISSUE_DYNAMIC_PATTERN = "dynamic_pattern"
# Patterns where a dependency is resolved from a runtime value, so cash cannot
# see an edit to it and a cached result can go silently stale: eval/exec/compile,
# getattr(obj, name)() dynamic dispatch, importlib.import_module. Caching
# correctness cannot be guaranteed, so these RAISE by default (opt in with
# assume_safe=True). Distinct from ISSUE_DYNAMIC_PATTERN, which stays advisory.
ISSUE_UNTRACKABLE_DEP = "untrackable_dep"
ISSUE_DISCARDED_CALL = "discarded_call"
ISSUE_SCOPE_MUTATION = "scope_mutation"
ISSUE_MUTABLE_GLOBAL = "mutable_global"
# Reading ambient state -- the clock, the environment, the working directory, a
# fresh UUID. Not a side effect: nothing about the world changes. The result
# depends on something the cache key cannot see, so the FIRST call's answer is
# what every later call gets, in this process and every process after it.
# Advisory like ISSUE_IMPURE_CALL (warn, still cache), because freezing is
# sometimes exactly what the user wants -- but it is never what they want by
# accident, and it is invisible without this.
ISSUE_AMBIENT_READ = "ambient_read"
# Fetching from a server or querying a database: `requests.get(url)`,
# `cur.execute("SELECT ...")`, `pd.read_sql(...)`. Also not a side effect -- the
# hazard is that the server's answer is an input the key cannot see, so the
# first answer is served until something changes the key. Unlike the clock,
# there is a knob made for exactly this: `ttl=` bounds how old a served answer
# may be, and setting one silences the advisory.
ISSUE_NETWORK_READ = "network_read"


@dataclass(frozen=True)
class PurityIssue:
    """A single issue surfaced by :class:`PurityAnalyzer`.

    Attributes:
        kind: One of ``impure_call``, ``dynamic_pattern``,
            ``untrackable_dep``, ``discarded_call``, ``scope_mutation``,
            ``mutable_global``, ``ambient_read``, ``network_read``.
        description: Human-readable summary (e.g. ``"requests.post()"``,
            ``"global X"``, ``"discards return of helper(...)``).
        where: Qualified name + line of the function containing the
            issue. For helpers, this is the helper's qualname so the
            user can fix the source of the problem, not just the
            outermost decorator.
        line: Line of the issue in the FILE that defines the function --
            what an editor's go-to-line takes. 0 for a finding about the
            whole function.
        filename: That file, so a finding in a helper names the helper's
            module rather than the file of the call that surfaced it.
        subject: The name the finding is about, where there is one -- the
            global of a ``mutable_global``.
        effect_kind: The :class:`~cash.effects.EffectKind` of the call an
            ``impure_call`` is about, where it has one.
    """

    kind: str
    description: str
    where: str
    line: int = 0
    filename: str = ""
    subject: str = ""
    effect_kind: EffectKind | None = None


@dataclass(frozen=True)
class PurityReport:
    """Result of analyzing a callable + its module-bounded helpers.

    Attributes:
        issues: All findings, in stable order (kind, line).
        helper_source_hashes: ``qualname -> digest`` for every user-code
            helper actually walked, captured at analysis time. Used as the
            fallback when per-call re-resolution fails (helper was
            deleted/renamed since analysis). The digest is over NORMALIZED
            source, so a comment or reformat in a helper does not
            invalidate its callers; for a helper with no readable source
            it is ``bytecode_identity``, matching what the per-call rehash
            computes for the same object.
        helper_resolution_paths: ``qualname -> (module_name, attr_chain)``
            for every walked helper. Used by the decorator to
            re-resolve the helper from ``sys.modules`` on each
            call and re-hash its source, so in-process
            redefinitions (notebook cells, REPL) invalidate the
            parent's cache key. The path is the one its CALLER uses:
            the caller's module and the name or attribute chain written
            at the call site (``("app", ("_sieve",))`` for
            ``from sievelib import sieve as _sieve``), so rebinding that
            name -- ``monkeypatch``, ``mock.patch`` -- reaches the key.
            Only a helper reached some other way (a class body's calls)
            falls back to its own home: ``(module, qualname chain)``.
        helper_bindings: ``(module_name, attr_chain, ref)`` for every
            call-site binding the walk followed -- helpers, cached callees
            and mocks alike -- where ``ref()`` is the object it held when
            analysed. ``bindings_changed`` compares them per call; a
            binding that no longer holds that object means the tree below
            it is not the one analysed.
        unkeyable: One description per binding that held a mock when
            analysed. A mock has no code to key and its answer is whatever
            the test configured, so a call reaching one runs uncached.
        opaque_callees: Qualified names of callees we encountered
            but couldn't read source for (C extensions, missing source,
            partial application). Treated as pure by default; strict
            mode promotes their presence to an issue.

            Opaque for PURITY only. One that still has a ``__code__``
            also gets an entry in ``helper_source_hashes`` and
            ``helper_resolution_paths``, digested from its compiled form,
            so editing it invalidates its callers.
    """

    issues: tuple[PurityIssue, ...] = ()
    helper_source_hashes: dict[str, str] = field(default_factory=dict)
    helper_resolution_paths: dict[str, tuple[str, tuple[str, ...]]] = field(default_factory=dict)
    #: ``qualname -> weakref`` for walked helpers that have NO resolution path
    #: -- a closure from a factory (``_make.<locals>.scaled``) cannot be looked
    #: up by qualname, so it used to be keyed by the analysis-time snapshot
    #: forever, which never sees its parameter defaults. Holding a
    #: weak reference lets the per-call rehash reach the live object.
    helper_objects: dict[str, Any] = field(default_factory=dict)
    opaque_callees: tuple[str, ...] = ()
    helper_bindings: tuple[tuple[str, tuple[str, ...], Any], ...] = ()
    #: ``(module_name, name)`` for each name a call site looks up in its
    #: module that the module had not bound when analysed: a builtin
    #: (``len``), or a helper defined further down the file than the first
    #: call. ``bindings_changed`` reports a change once the module binds one,
    #: so the function is analysed again and the new helper is keyed.
    unbound_names: tuple[tuple[str, str], ...] = ()
    unkeyable: tuple[str, ...] = ()
    #: Binding paths every call site of which is on a ``# @cash:assume-safe``
    #: line (``LEDGER.record(r)  # @cash:assume-safe``). The code is still
    #: followed; the data the bound object carries is not keyed, because the
    #: audited effect is what moves it (a ledger's count, a client's stats).
    waived_bindings: frozenset[tuple[str, tuple[str, ...]]] = frozenset()
    #: Environment reads, in the function and its helpers, whose current value
    #: the key folds on every call: ``("env", NAME)`` or ``("cwd", "")``
    #: (`cash.effects.environment_input`).
    environment_reads: frozenset[tuple[str, str]] = frozenset()
    #: A reference (``ref()``) to each ``@cash.cache`` wrapper the walk met, in
    #: the function or in any helper it reaches: called, imported in the body,
    #: or named as a value. Not walked -- each is a graph edge of its own --
    #: but an edge only its registry can make: a cached function reached
    #: through a plain helper or a function-local import had none, and an
    #: edit to it served its callers' old results.
    cached_callees: tuple[Any, ...] = ()
    #: Why the walk for the key could not finish (`HelperWalk.WALK_LIMIT`),
    #: or "". A report that did not reach every helper cannot key the call,
    #: so the call runs uncached.
    unwalkable: str = ""

    @property
    def is_clean(self) -> bool:
        """True when no issues were flagged."""
        return not self.issues

    def format(self) -> str:
        """Human-readable multi-line summary."""
        if self.is_clean:
            return "(no purity issues detected)"
        by_where: dict[str, list[PurityIssue]] = {}
        for issue in self.issues:
            by_where.setdefault(issue.where, []).append(issue)
        lines = []
        for where, issues in by_where.items():
            lines.append(f"  in {where}{_file_part(issues)}:")
            for i in issues:
                lines.append(f"    line {i.line}: [{i.kind}] {i.description}")
        return "\n".join(lines)


def _file_part(issues: list[PurityIssue]) -> str:
    """`` (path/to/file.py)`` for a group of issues, or nothing."""
    name = next((i.filename for i in issues if i.filename), "")
    return f" ({name})" if name else ""

"""What the decorator does with the purity analyzer's findings and with the
effects it observes while a body runs."""

from __future__ import annotations

import ast
import dis
import inspect
import logging
import textwrap
import types
from collections.abc import Callable
from typing import Any

from .. import _plain_data
from .._clock import perf_counter as _perf_counter
from .._paths import MAIN_MODULE_NAMES, resolve_main_module
from ..analysis.cacheability_decision import identity_coupled_reason
from ..effect_observer import EffectObserver, observed_label
from ..exceptions import (
    SOURCE_RETRIEVAL_ERRORS,
    CashCacheIneffectiveWarning,
    CashImpureFunctionError,
    CashImpurityWarning,
)
from ..purity_analyzer import (
    ISSUE_AMBIENT_READ,
    ISSUE_IMPURE_CALL,
    ISSUE_MUTABLE_GLOBAL,
    ISSUE_NETWORK_READ,
    ISSUE_UNTRACKABLE_DEP,
    PurityIssue,
    PurityReport,
    resolve_binding,
)
from ..value_types import IMMUTABLE_VALUE_TYPES, writable_types
from .closure_fold import is_immutable_capture, iter_code_scopes, unsafe_uses_of
from .code_identity import func_key, is_user_module, own_package
from .globals_fold import stabilize_for_global_hash

logger = logging.getLogger(__name__)


def is_mutable(value) -> bool:
    """Whether a caller can write through *value*, so a copy would differ.

    Only what can actually be written into: a container, an array or a frame,
    or an object whose attributes can be rebound (it has a ``__dict__``). A
    `date`, a `Path`, a `Decimal`, a string or a number cannot be changed
    through the name at all, so handing back a copy of one is the same value --
    warning about those made `return sum(rows), as_of` a finding.
    """
    if isinstance(value, writable_types()):
        return True
    return getattr(type(value), "__dictoffset__", 0) != 0 and hasattr(value, "__dict__")


def shares_memory(result, value) -> bool:
    """Whether *result* and *value* may sit on the same buffer, cheaply.

    ``may_share_memory`` is a bounds check, not the exact analysis, so it costs
    nothing and errs toward saying yes -- which for a warning is the right
    direction.
    """
    try:
        import numpy as _np
    except ImportError:
        return False
    if not isinstance(result, _np.ndarray) or not isinstance(value, _np.ndarray):
        return False
    return bool(_np.may_share_memory(result, value))


def make_opaque_issue(func_name: str, opaque_list: str) -> Any:
    """Build a synthetic `PurityIssue` for opaque callees
    encountered in ``strict`` mode. Defined at module scope so the
    ``_surface_purity`` import stays local."""

    return PurityIssue(
        kind=ISSUE_IMPURE_CALL,
        description=f"opaque callees (strict): {opaque_list}",
        where=func_name,
        line=0,
    )


def format_issues_summary(issues: list[Any]) -> str:
    """Pretty-print a list of `PurityIssue` records, grouped
    by their ``where`` field. Used by both the warning body and the
    strict-mode exception body so users get the same diagnostic.
    """
    by_where: dict[str, list[Any]] = {}
    for i in issues:
        by_where.setdefault(i.where, []).append(i)
    lines = []
    for where in sorted(by_where):
        # The defining file, so a finding in a helper names the helper's
        # module -- the warning's own header names the CALL site's file.
        filename = next((getattr(i, "filename", "") for i in by_where[where] if getattr(i, "filename", "")), "")
        lines.append(f"  in {where} ({filename}):" if filename else f"  in {where}:")
        for issue in by_where[where]:
            line_part = f"line {issue.line}: " if issue.line else ""
            lines.append(f"    {line_part}[{issue.kind}] {issue.description}")
    return "\n".join(lines)


def static_effect_kinds(report: Any) -> set[str]:
    """The observed-effect kinds (``EffectObserver``) a static report names.

    So an effect the static warning already listed is not repeated by
    IMPURE-OBSERVED-EFFECTS, while one of another kind still is. Errs toward
    NOT covering: a finding with no effect kind covers nothing.
    """
    kinds: set[str] = set()
    for issue in getattr(report, "issues", ()) or ():
        if "changes the argument" in getattr(issue, "description", ""):
            kinds.add("argument mutation")
        if getattr(issue, "kind", None) not in (ISSUE_IMPURE_CALL, ISSUE_NETWORK_READ):
            continue
        label = observed_label(getattr(issue, "effect_kind", None))
        if label is not None:
            kinds.add(label)
    return kinds


#: How deep into a returned container an argument is looked for.
SHARED_RESULT_DEPTH = 2


def describe_scope_use(reader: Any, name: str, cached: Any) -> str:
    """`` -- through `X.f()` in mod.helper (file:line)`` for the warning, or ``""``.

    *reader* is the function whose read of *name* was watched: the cached
    function, or a helper several calls below it. The move itself may be
    deeper still (a method of the object), but this is the line in code the
    user wrote that reaches it -- the first use that may mutate, the same
    rule that made the name provisional. Only runs when warning.
    """
    if reader is None:
        return ""
    try:
        lines, first = inspect.getsourcelines(reader)
        filename = inspect.getsourcefile(reader) or inspect.getfile(reader)
        source = textwrap.dedent("".join(lines))
        tree = ast.parse(source)
    except (*SOURCE_RETRIEVAL_ERRORS, SyntaxError):
        return ""
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Call, ast.Assign, ast.AugAssign, ast.Delete)):
            continue
        if name not in unsafe_uses_of(node, {name}):
            continue
        if best is None or (node.lineno, node.col_offset) < (best.lineno, best.col_offset):
            best = node
    if best is None:
        return ""
    lineno = max(first, 1) - 1 + best.lineno
    text = " ".join((ast.get_source_segment(source, best) or "").split())
    if len(text) > 80:
        text = text[:77] + "..."
    inside = "" if reader is cached else f" in {func_key(reader)}"
    return f" -- through `{text}`{inside} ({filename}:{lineno})"


#: A re-hash costing more than this marks the function as not worth
#: verifying again. Measured: ~7.4 ms for a 16 MB ndarray, so this is
#: roughly a 100 MB argument. The first miss still gets checked -- the
#: budget only stops a large argument from being re-hashed on every
#: subsequent miss.
MUTATION_CHECK_BUDGET_S = 0.05


def helper_mutates_global(fn: Any, name: str) -> bool:
    """Does *fn*'s own body rebind *name* or change it in place?"""
    code = getattr(fn, "__code__", None)
    if code is None:
        return False

    for scope in iter_code_scopes(code):
        for instr in dis.get_instructions(scope):
            if instr.opname in ("STORE_GLOBAL", "DELETE_GLOBAL") and instr.argval == name:
                return True
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except SOURCE_RETRIEVAL_ERRORS + (SyntaxError,):
        return False
    return name in unsafe_uses_of(tree, frozenset({name}), bare_args=False, mutating_methods_only=True)


class PurityChecksMixin:
    """Purity findings, observed effects and argument mutation, per cached function."""

    def _warn_shared_result(self, func, func_name: str, result, args, kwargs) -> None:
        """Say so when the result shares state with something the caller holds.

        A hit hands back a value rebuilt from the stored bytes, so what the
        computing run shares, a later run does not: ``return base[lo:hi]``
        stops sharing memory with ``base``, ``return Wrapper(rows)`` stops
        holding the caller's list, and ``return CONFIG`` stops carrying the
        caller's writes back to the module (correct on the computing run,
        silently different on every later one).

        Cash cannot tell whether the caller relies on that sharing, so this
        names what will differ rather than refusing to cache; ``assume_safe``
        waives it. Detection is cheap and evidence-only: memory shared with an
        ndarray argument (a bounds check), the result BEING or holding an
        argument (identity), or the result being one of the function's module
        globals (identity).
        """
        if self._purity_mode(func_name) == "silent":
            return
        try:
            shared = self._shared_with(result, args, kwargs, func)
        except Exception:  # noqa: BLE001 - a diagnostic must never break a call
            return
        if shared is None:
            return
        what, name = shared
        self._warn_once(
            CashImpurityWarning,
            func_name,
            "shared-result",
            f"@cash.cache on {func_name}: the result {what} '{name}', which the "
            f"caller still holds. A cache HIT hands back a value rebuilt from "
            f"the stored bytes, so from the next run on they are separate "
            f"objects: writes through one will not be seen in the other.",
            code="CACHE-RESULT-SHARED",
            fix=(
                "return something of its own (`.copy()`, `dict(...)`, "
                "`list(...)`) if the caller reads it independently; pass "
                "assume_safe=True once you have checked that nothing relies "
                "on the sharing."
            ),
        )

    def _shared_with(self, result, args, kwargs, func) -> tuple[str, str] | None:
        """``(what, name)`` for the first sharing found in *result*, or None."""

        try:
            names = list(inspect.signature(func).parameters)
        except (TypeError, ValueError):
            names = []
        supplied = [(names[i] if i < len(names) else f"arg{i}", value) for i, value in enumerate(args)]
        supplied += list(kwargs.items())

        def contains(value, target, depth):
            if value is target:
                return True
            if depth <= 0:
                return False
            kind = type(value)
            if kind is dict:
                return any(contains(v, target, depth - 1) for v in value.values())
            if kind in (list, tuple, set, frozenset):
                return any(contains(v, target, depth - 1) for v in value)
            state = getattr(value, "__dict__", None)
            if isinstance(state, dict):
                return any(contains(v, target, depth - 1) for v in state.values())
            return False

        for name, value in supplied:
            if not is_mutable(value):
                # Nothing can be written through it, so nothing can differ.
                continue
            if value is result:
                return "is the argument", name
            if contains(result, value, SHARED_RESULT_DEPTH) and is_mutable(value):
                return "holds the argument", name
            shared = shares_memory(result, value)
            if shared:
                return "shares memory with the argument", name
        globals_ = getattr(func, "__globals__", None)
        if isinstance(globals_, dict):
            for name, value in list(globals_.items()):
                if value is result and is_mutable(value):
                    return "is the module global", name
        return None

    def _learn_mutating_captures(self, func: Callable, func_name: str, watched: dict[str, tuple[str, str]]) -> None:
        """Demote any provisional global this call was OBSERVED to mutate.

        A global merely *passed to a call* (`sum(G)`, `model.predict(G)`) might
        be mutated by the callee, but dropping it from the key would serve
        stale values forever. Such names are folded, and confirmed here: hash
        them again once the body has run and compare against the hash the key
        already needed.

        Changed across the call => calling this function is what moves the value,
        so folding it would key the entry on the function's own output and miss
        forever. Stop folding that ONE name; the function keeps caching on
        everything else.

        Two things worth knowing:

        * The entry just written stays valid -- it is keyed on the PRE-call
          state, which is what produced it. The next call keys without this
          name, misses once, and thereafter keys like any watched global.
        * A change *between* calls (`G = [...]` anywhere) is invisible to this
          window by construction, which is correct: that is precisely what
          folding is for, and it needs no detection.

        Only runs on a miss -- the body has to execute for there to be anything
        to observe -- so the cost is one hash on the path that just paid for a
        real computation.
        """
        if not watched:
            return
        code = getattr(func, "__code__", None)
        if code is None:
            return
        own_globals = getattr(func, "__globals__", None)
        cells = dict(zip(getattr(code, "co_freevars", ()) or (), getattr(func, "__closure__", ()) or ()))
        for name, (before, scope, owner, reader) in watched.items():
            try:
                if scope == "closure":
                    cell = cells.get(name)
                    if cell is None:
                        continue
                    after = self._hash_arg_payload((cell.cell_contents,), {})
                elif scope == "carrier":
                    mapping, key = owner
                    if key not in mapping:
                        continue
                    after = self._carried_global_hash(mapping[key], getattr(func, "__module__", None))
                else:
                    # The mapping the BEFORE hash came from -- a helper's
                    # module, when this entry was folded on a helper's behalf.
                    g = owner if isinstance(owner, dict) else own_globals
                    if not isinstance(g, dict) or name not in g:
                        continue
                    after = self._hash_arg_payload(
                        (stabilize_for_global_hash(g[name], self._data_callable_identity),), {}
                    )
            except Exception:  # noqa: BLE001 - unhashable NOW; treat as unchanged
                continue
            if after == before:
                continue
            if scope == "carrier":
                # The library's own state (a generator advanced, a cache
                # filled), not a mutation the user wrote: stop folding it.
                self._mutating_globals.setdefault((code, "global"), set()).add(name)
                logger.debug("[CORE] %s: stopped keying what %s carries; calling it changes it", func_name, name)
                continue
            self._mutating_globals.setdefault((code, scope), set()).add(name)
            if scope == "closure":
                where = f"variable it captures '{name}'"
            else:
                module = owner.get("__name__") if isinstance(owner, dict) else None
                if module in MAIN_MODULE_NAMES and reader is not None:
                    module = resolve_main_module(reader)
                where = f"module global '{module}.{name}'" if module else f"module global '{name}'"
            site = describe_scope_use(reader, name, func)
            self._warn_once(
                CashImpurityWarning,
                func_name,
                name,
                f"@cash.cache on {func_name}: calling it modifies the {where}"
                f"{site}, so '{name}' can no longer be tracked for "
                f"invalidation and a cache hit will not repeat that change.",
                code="IMPURE-SCOPE-MUTATION",
                fix="pass the value in as an argument and return the new one, "
                "instead of reaching out and rewriting it; or, if a cache hit "
                "may skip the change (a bill, a log), put "
                "`# @cash:assume-safe` on the line named.",
            )

    def _refuses_identity_coupled(self, func_name: str, result: Any) -> bool:
        """True when *result* must never be stored, because storing it would
        detach a library's global registry from the object the caller holds.

        The statement path (``statement/processor.py``) and call interception
        (``call_unit.py``) have gated on ``identity_coupled_reason`` for a
        while; the decorator did not.  So ``@cash.cache`` on a function
        returning a ``Figure`` hijacked ``plt.gcf()`` -- on the FIRST call,
        during the *store*, because the RAM tier deep-copies and
        ``Figure.__setstate__`` re-registers the copy as pyplot's current
        figure.  The user then draws on their figure while ``plt.savefig()``
        writes the cache's private snapshot.

        Checked here rather than inside ``_store_in_cache`` so the refusal
        lands beside ``cache_if``, BEFORE ``_attach_lineage``: a value that is
        not stored must not carry a lineage hash pointing at an entry that was
        never written.

        KNOWN BOUNDARY: called at all four store sites (sync/async x
        non-iterator/single-chunk), which is every site where the value is in
        hand before anything is written.  A *multi*-chunk iterator is not
        covered -- ``_stream_and_store`` has already written earlier chunks by the
        time any item could be inspected, so gating there would mean aborting
        mid-write and reclaiming them.  Reaching it needs a generator yielding
        enough Figures to cross ``chunk_max_bytes`` (or a million of them),
        which no reported case comes near.  Widen this if one ever does.
        """

        # ``func_name`` is already in the message prefix, so name the slot
        # rather than repeating the qualified path inside the reason.
        reason = identity_coupled_reason("the returned value", result)
        if reason is None:
            return False
        self._warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "",
            f"@cash.cache on {func_name}: result not cached. {reason}",
            code="CACHE-IDENTITY-COUPLED",
            fix="split the function: cache the part that computes the numbers, "
            "and draw the figure from them in an uncached function.",
        )
        return True

    def _argument_snapshot(self, func_name: str, args: tuple, kwargs: dict) -> dict[str, str] | None:
        """``{parameter: hash}`` of the arguments that CAN change, before the body.

        An int, a str, a tuple of them: rebinding one inside the body (``n -=
        1``) is invisible to the caller, so they are left out, and most calls
        snapshot nothing. What remains lets `_check_argument_mutation` name the
        argument that moved. None when the check has been retired as too
        costly for this function, or nothing could be hashed.
        """
        cf = self._cached.get(func_name)
        if cf is None or cf.mutation_check_retired:
            return None
        # The key was hashed a moment ago, on this thread: if that already cost
        # more than the check may, the check is retired before it pays -- a
        # miss on two million rows hashed them three times, once for the key,
        # once here and once after the body. Read from what
        # `_note_arg_cost` kept: it has already taken `ARG_COST.last`.
        cost = cf.arg_cost
        if cost is not None and cost[2] > MUTATION_CHECK_BUDGET_S:
            cf.mutation_check_retired = True
            return None
        started = _perf_counter()
        try:
            canon_args, canon_kwargs = self._normalize_call_args(func_name, args, kwargs)
        except Exception:  # noqa: BLE001 - best effort, like the check itself
            return None
        named = [(f"*args[{i}]", v) for i, v in enumerate(canon_args)] + list(canon_kwargs.items())
        snapshot: dict[str, str] = {}
        immutable = IMMUTABLE_VALUE_TYPES
        candidates = [
            (name, value) for name, value in named if not (is_immutable_capture(value) or isinstance(value, immutable))
        ]
        if len(candidates) == 1:
            # The only argument that can change is the one that did: named by
            # elimination, with no hash of its own.
            return {candidates[0][0]: ""}
        for name, value in candidates:
            try:
                snapshot[name] = self._hash_arg_payload((value,), {})
            except Exception:  # noqa: BLE001 - unhashable: the whole-args check still runs
                continue
        if _perf_counter() - started > MUTATION_CHECK_BUDGET_S:
            cf.mutation_check_retired = True
        return snapshot

    def _argument_identities(self, func_name: str, args: tuple, kwargs: dict) -> dict[str, tuple[Any, list]]:
        """``{parameter: (value, identity snapshot)}`` for the plain lists and
        tuples a call receives, before the body runs.

        The hash snapshot below is retired for a big argument and never covers
        a frozen one, which is exactly where ``rows.sort()`` on a million
        parsed rows, or a field rewritten in every row of a frozen result, got
        stored. Identities cost a fraction of a hash, so they are
        taken whatever the size (`_plain_data.identity_snapshot`).
        """
        try:
            canon_args, canon_kwargs = self._normalize_call_args(func_name, args, kwargs)
        except Exception:  # noqa: BLE001 - best effort, like the check itself
            return {}
        found: dict[str, tuple[Any, list]] = {}
        named = [(f"*args[{i}]", v) for i, v in enumerate(canon_args)] + list(canon_kwargs.items())
        for name, value in named:
            if type(value) is list or type(value) is tuple:
                snapshot = _plain_data.identity_snapshot(value)
                if snapshot is not None and any(level is not None for level in snapshot):
                    found[name] = (value, snapshot)
        return found

    def _check_argument_mutation(
        self,
        func_name: str,
        args: tuple,
        kwargs: dict,
        args_hash: str | None,
        observer: EffectObserver | None,
    ) -> None:
        """Did the body change the arguments it was handed?

        The static analyzer sees `rows.append(x)` written in the function, but
        not `vendor.normalize(rows)` sorting the list inside a library it does
        not walk into. Measured: of four state mutations that reached past the
        analyzer, three left an observable change in the arguments -- so the
        cheapest way to find them is to look.

        The argument hash is already computed to build the cache key, so this
        re-runs exactly that and compares. `_serialize_args` canonicalises
        (kwargs order included) and is deterministic on unchanged input, which
        is what makes a difference mean *mutation* rather than noise.

        Runs only on a miss, and only while it stays cheap: a re-hash over the
        budget above retires the check for that function rather than taxing
        every later miss.
        """
        if args_hash is None or observer is None:
            return
        identities = observer.arg_identities
        if identities:
            moved = [
                name for name, (value, snapshot) in identities.items() if _plain_data.identity_changed(value, snapshot)
            ]
            if moved:
                for name in moved:
                    self._forget_frozen_container(identities[name][0])
                observer.mutated_args = moved
                observer.record(
                    "argument mutation",
                    f"the call changed {', '.join(repr(n) for n in moved)} in place -- "
                    f"the result was not stored, so this call runs every time",
                )
                return
        cf = self._cached.get(func_name)
        if cf is None or cf.mutation_check_retired:
            return
        started = _perf_counter()
        try:
            after = self._serialize_args(func_name, args, kwargs)
        except Exception:  # noqa: BLE001 - user arguments' hashing
            # Hashing is best-effort here. An argument that hashed once and
            # not twice (a generator drained by the body, say) is not evidence
            # of mutation, and must not be reported as such.
            return
        if _perf_counter() - started > MUTATION_CHECK_BUDGET_S:
            cf.mutation_check_retired = True
        if after is None or after == args_hash:
            return
        names = self._mutated_argument_names(func_name, args, kwargs, observer)
        observer.mutated_args = names or ["an argument"]
        shown = ", ".join(repr(n) for n in names) if names else "an argument"
        observer.record(
            "argument mutation",
            f"the call changed {shown} in place -- the result was not stored, so this call runs every time",
        )

    def _mutated_argument_names(self, func_name: str, args: tuple, kwargs: dict, observer: EffectObserver) -> list[str]:
        """The parameters whose value moved across the call, by name."""
        before = observer.arg_snapshot
        if not before:
            return []
        if len(before) == 1:
            return list(before)
        now = self._argument_snapshot(func_name, args, kwargs) or {}
        return [name for name, digest in before.items() if now.get(name) != digest]

    def _make_effect_observer(self) -> EffectObserver:
        """An :class:`EffectObserver` scoped to this instance's cache dir.

        Excluding the cache directory is load-bearing: cash writes the entry
        it is computing, and without the exclusion every cached function would
        be observed writing a file and every one of them would warn.
        """

        cache_dir = getattr(self.config, "cache_dir", None)
        return EffectObserver(exclude_under=cache_dir)

    def _report_observed_effects(self, func_name: str, observer: EffectObserver | None) -> None:
        """Warn once when the first call did something a hit will not do.

        Silent when:

        * ``assume_safe=True`` -- the user audited this function and said so;
          ``# @cash:assume-safe`` on a line that led to an effect waives that
          effect alone (see ``EffectObserver.record_effect``).
        * the static findings already name that KIND of effect -- a write the
          analyzer listed is not news when the observer sees it too. Only the
          kinds they cover are dropped, so a static finding about a log line
          does not hide a network read in the same function.
        * nothing was observed -- which is *not* proof of purity. Only the
          path this call took was watched, so an effect behind a branch that
          did not run is unobserved. That is why this supplements the static
          pass rather than replacing it.
        """
        if observer is None or not observer.effects:
            return
        if self._purity_mode(func_name) == "silent":
            return
        covered: set[str] = set()
        if func_name in self._purity_static_flagged:
            covered = static_effect_kinds(self._purity_reports.get(func_name))
        effects = [(kind, detail) for kind, detail in observer.effects if kind not in covered]
        if not effects:
            return
        summary = "\n".join(dict.fromkeys(f"  {kind}: {detail}" for kind, detail in effects))
        self._warn_once(
            CashImpurityWarning,
            func_name,
            "observed_effect",
            f"@cash.cache on {func_name}: the first call had effects that "
            f"static analysis did not see -- they happen inside library code "
            f"cash does not walk into. Every cache HIT from here on returns "
            f"the stored value WITHOUT repeating them.\n{summary}",
            code="IMPURE-OBSERVED-EFFECTS",
            fix="split the function if an effect is part of the result -- an "
            "'argument mutation' line means an object the CALLER still "
            "holds stops being changed. If it is incidental, put "
            "`# @cash:assume-safe` on the line named (any line of yours on "
            "the way to it counts); @cash.cache(assume_safe=True) waives "
            "the whole function instead, including effects added later.",
        )

    def _mutable_global_is_keyed(self, func_name: str, report: PurityReport, issue: Any) -> bool:
        """Is this ``mutable_global`` finding about a global the key already
        folds by value on every call?

        Then "cached results won't reflect changes to it" is false: a setter
        rebinding it, or a test patching it, makes the next call a new entry.
        Measured: a `configure()`-set module flag re-ran the
        function each time it changed, 0 diffs against a no-cache oracle --
        while the warning, of the kind the docs say never to ignore, said
        otherwise. Kept for what the fold leaves out: callables, modules,
        classes (tracked their own way, or not at all), and a global the
        function itself writes.
        """

        name = getattr(issue, "subject", "")
        if getattr(issue, "kind", None) != ISSUE_MUTABLE_GLOBAL or not name:
            return False
        func = self.functions.get(func_name)
        reader: Any = func
        where = getattr(issue, "where", "")
        if where in report.helper_resolution_paths:
            reader = resolve_binding(*report.helper_resolution_paths[where])
        elif where in report.helper_objects:
            reader = report.helper_objects[where]()
        module_ns = getattr(reader, "__globals__", None)
        if not isinstance(module_ns, dict) or name not in module_ns:
            return False
        if reader is not func and helper_mutates_global(reader, name):
            # The loader of a lazily filled settings dict (`_CFG.clear();
            # _CFG.update(...)`) "reads" it only to fill it: its writes are
            # findings of their own, and the dict is not its input. Reported
            # as a stale-result risk, it was one more false alarm on the most
            # common settings pattern there is.
            return True
        try:
            if name not in self._read_global_data_names(reader):
                return False
        except Exception:  # noqa: BLE001 - user source; a heuristic must not break a call
            return False
        value = module_ns[name]
        if isinstance(value, types.ModuleType):
            # `conf.RATE` reads of a module of the user's are folded by value
            # (`_module_attr_parts`); "mutated elsewhere" is `conf.RATE = ...`.
            return is_user_module(value, own_package(reader))
        if isinstance(value, type):
            return False
        return not (callable(value) and not isinstance(value, (dict, list, tuple, set)))

    def _surface_purity(
        self,
        func_name: str,
        report: PurityReport,
        mode: str,
    ) -> None:
        """Turn a `PurityReport` into warnings or an exception.

        Called once per function on first call (after first
        ``_analyze_dependencies``).

        * ``warn`` (default): one-shot `CashImpurityWarning`
          summarising issues; also recorded in
          ``cache_info()['warnings']``.
        * ``silent`` (``assume_safe=True``): the user has audited
          this; suppress the warning. The report is still stored
          so helper source hashes invalidate correctly.
        * ``strict``: raise `CashImpureFunctionError`. Opaque
          callees count as issues in this mode (paranoid).
        """
        issues = [i for i in report.issues if not self._mutable_global_is_keyed(func_name, report, i)]
        if any(getattr(i, "kind", None) == ISSUE_NETWORK_READ for i in issues):
            # Named statically, so the observer does not report the same read
            # as a connection -- whether or not the advisory below is shown.
            self._purity_static_flagged.add(func_name)
            cf = self._cached.get(func_name)
            if self._effective_ttl(func_name, cf.ttl if cf is not None else None) is not None:
                # `ttl=` is the answer to "how old may a fetched answer be":
                # once one is set, the question has been answered.
                issues = [i for i in issues if getattr(i, "kind", None) != ISSUE_NETWORK_READ]
        if mode == "strict" and report.opaque_callees:
            opaque_list = ", ".join(report.opaque_callees[:5])
            if len(report.opaque_callees) > 5:
                opaque_list += f", ... +{len(report.opaque_callees) - 5} more"
            issues.append(make_opaque_issue(func_name, opaque_list))

        if not issues:
            return
        if mode == "silent":
            return

        summary = format_issues_summary(issues)

        # Untrackable-dependency patterns (eval/exec/compile, getattr(obj,name)()
        # dynamic dispatch, importlib.import_module) RAISE by default, even in
        # the ordinary "warn" mode: cash cannot see an edit to a dependency it
        # resolves from a runtime value, so a cached result can go silently
        # stale, and caching correctness can no longer be guaranteed. The user
        # must acknowledge the risk with assume_safe=True (the ``silent`` mode
        # handled above) to cache anyway.
        untrackable = [i for i in issues if getattr(i, "kind", None) == ISSUE_UNTRACKABLE_DEP]
        if untrackable and mode != "strict":
            untrackable_summary = format_issues_summary(untrackable)
            raise CashImpureFunctionError(
                f"@cash.cache on {func_name}: a dependency is resolved from a "
                f"runtime value, so cash cannot tell when it changes and a cached "
                f"result could be silently stale. Caching correctness cannot be "
                f"guaranteed for this function.\nPut `# @cash:assume-safe` on "
                f"the line named below to accept the risk for that statement "
                f"alone, pass @cash.cache(assume_safe=True) to waive the whole "
                f"function, or refactor to a statically-named "
                f"call.\n{untrackable_summary}"
            )

        # Ambient reads get their own warning, not the side-effects one. The
        # hazard is the opposite shape -- nothing is skipped, a hidden INPUT is
        # frozen -- and so is the fix: pass the value in as an argument, where
        # it reaches the key. Filing them under "likely side effects" told the
        # user to audit for writes that are not there, and left the actual
        # failure (a nightly job whose `date.today()` is the night it first
        # ran) unnamed.
        # strict=True keeps them in the one exception it raises: there, every
        # issue is a hard stop and splitting the report would hide half of it.
        ambient = [i for i in issues if getattr(i, "kind", None) == ISSUE_AMBIENT_READ]
        if ambient and mode != "strict":
            issues = [i for i in issues if getattr(i, "kind", None) != ISSUE_AMBIENT_READ]
            self._purity_static_flagged.add(func_name)
            self._warn_once(
                CashImpurityWarning,
                func_name,
                "ambient",
                f"@cash.cache on {func_name}: the body reads ambient state "
                f"(the clock, the environment, the working directory, a fresh "
                f"UUID). That value is not part of the cache key, so the first "
                f"call's answer is what every later call gets back -- in this "
                f"process and in every process "
                f"after it.\n{format_issues_summary(ambient)}",
                code="KEY-AMBIENT-READ",
                fix="pass the value in as an argument -- `f(now=datetime.now())` "
                "-- so it reaches the cache key and a new value means a new "
                "entry. If freezing it is what you want, say so with "
                "`# @cash:assume-safe` on that line.",
                once_per_version=True,
            )
        # A network or database read gets its own advisory too, for the same
        # reason as an ambient read: nothing is skipped, an input the key
        # cannot see is frozen. Unlike the clock it has a knob made for it,
        # `ttl=`, which silences it (above). Under strict=True it raises with
        # the other issues unless a ttl= is set.
        remote = [i for i in issues if getattr(i, "kind", None) == ISSUE_NETWORK_READ]
        if remote and mode != "strict":
            issues = [i for i in issues if getattr(i, "kind", None) != ISSUE_NETWORK_READ]
            self._warn_once(
                CashImpurityWarning,
                func_name,
                "network_read",
                f"@cash.cache on {func_name}: the result depends on what a "
                f"server or a database returned, and that answer is not part "
                f"of the cache key. The first call's answer is what every later call gets "
                f"back -- in this process and in every process after it -- "
                f"until something changes the key.\n{format_issues_summary(remote)}",
                code="KEY-NETWORK-READ",
                fix="bound how old a served answer may be with ttl= -- "
                "`@cash.cache(ttl=3600)` -- or pass what makes the answer new "
                "(a date, a version) as an argument, so it reaches the key. If "
                "the answer never changes, say so with `# @cash:assume-safe` "
                "on that line.",
                once_per_version=True,
            )
        if not issues:
            return
        summary = format_issues_summary(issues)

        self._purity_static_flagged.add(func_name)

        if mode == "strict":
            raise CashImpureFunctionError(
                f"@cash.cache(strict=True) on {func_name}: purity issues "
                f"detected. Fix the function, mark callees with "
                f"@pure / @stateful, put `# @cash:assume-safe` on the lines you "
                f"have audited, or relax to assume_safe=True.\n{summary}"
            )
        # mode == "warn"
        self._warn_once(
            CashImpurityWarning,
            func_name,
            "purity",
            f"@cash.cache on {func_name}: reading the source found likely "
            f"side effects or scope mutations, so cached results may not "
            f"reflect what the body does.\n{summary}",
            code="IMPURE-SIDE-EFFECTS",
            fix=(
                (
                    "for a line that changes an argument in place, return a "
                    "modified copy instead -- the caller keeps its object "
                    "whether the call hits or misses. "
                    if "changes the argument" in summary or "element of the argument" in summary
                    else ""
                )
                + "go down the list and put `# @cash:assume-safe` on each line "
                "you have audited, or refactor; @cash.cache(assume_safe=True) "
                "waives the whole function instead, including anything added "
                "to it later. The first annotation changes the function's key "
                "once: @cash: directives are part of its source identity."
            ),
            once_per_version=True,
        )

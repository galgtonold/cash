"""What the decorator does with the purity analyzer's findings and with the
effects it observes while a body runs."""

from __future__ import annotations

import ast
import dataclasses
import dis
import enum
import hashlib
import inspect
import logging
import sys
import textwrap
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .. import _plain_data, identity_refs
from .._clock import perf_counter as _perf_counter
from .._paths import MAIN_MODULE_NAMES, resolve_main_module
from ..analysis.cacheability_decision import identity_coupled_reason
from ..analysis.helper_bindings import resolve_binding
from ..analysis.purity_visitor import ARGUMENT_MUTATION_NOT_STORED
from ..analysis.purity_report import (
    ISSUE_AMBIENT_READ,
    ISSUE_IMPURE_CALL,
    ISSUE_MUTABLE_GLOBAL,
    ISSUE_NETWORK_READ,
    ISSUE_UNTRACKABLE_DEP,
    PurityIssue,
    PurityReport,
)
from ..effect_observer import EffectObserver, observed_label
from ..effects import EffectKind
from ..exceptions import (
    SOURCE_RETRIEVAL_ERRORS,
    CashCacheIneffectiveWarning,
    CashImpureFunctionError,
    CashImpurityWarning,
)
from ..install_paths import is_user_code_module, is_user_module
from ..source_reading import getsource, getsourcelines
from ..tracking.randomness import capture_argument_carrier_states, moved_carriers, rng_carrier_kind
from ..value_types import IMMUTABLE_VALUE_TYPES, writable_types
from .arg_key import keyed_arguments
from .cash_key import cash_key_method_of_type
from .closure_fold import capture_digest, iter_code_scopes, unsafe_uses_of
from .function_identity import func_key
from .key_values import is_immutable_capture, plain_data_kind
from .user_code import own_package

if TYPE_CHECKING:
    from ..config.schema import CashConfig
    from .arg_hashing import ArgHasher
    from .class_data import ClassDataFold
    from .frozen import FrozenResults
    from .global_reads import GlobalReads
    from .global_values import GlobalValues
    from .registry import FunctionRegistry
    from .reporting import Notices

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


def compared_by_identity(value) -> bool:
    """Whether *value* is only ever equal to itself (its type keeps ``object.__eq__``).

    A copy of such a value is a different value, so a hit that IS a module
    global or closure variable of the function hands back that variable's
    own object (`ResultStore.restore_identity`): ``MISSING = object()``. A
    class, function or module pickles by reference and needs none of this.
    """
    if value is None or isinstance(value, (type, types.ModuleType)) or inspect.isroutine(value):
        return False
    return type(value).__eq__ is object.__eq__


def sentinel_ref(func, result) -> list | None:
    """``["global" | "closure", name]`` when *func* returns that variable's own object.

    Only for a *result* compared by identity (`compared_by_identity`) that the
    body names itself: ``return d.get(k, MISSING)``. A global the result merely
    happens to be -- ``max(candidates)`` picking a module-level object passed in
    as an argument -- is not the function's sentinel: a hit handing back that
    variable would hand back whatever it holds then, an object the call never saw.
    """
    if not compared_by_identity(result):
        return None
    try:
        code = func.__code__
        for name, cell in zip(code.co_freevars, func.__closure__ or ()):
            if cell.cell_contents is result:
                return ["closure", name]
        named = {name for scope in iter_code_scopes(code) for name in scope.co_names}
        globals_ = func.__globals__
        for name in named:
            if globals_.get(name, _MISSING_GLOBAL) is result:
                return ["global", name]
    except (AttributeError, ValueError, RuntimeError):
        return None
    return None


_MISSING_GLOBAL = object()


def held_sentinels(func) -> dict[int, list]:
    """``id(object) -> ["global" | "closure", name]`` for each sentinel-like
    object *func*'s body names: a bare ``object()``, or an instance of the
    user's own class compared by identity (`compared_by_identity`).

    What a result may hold inside it and a hit must hand back as that same
    object (`cash.identity_refs`). Not a library's object (a logger, a
    lock): naming one is common, holding one in a result is not, and every
    store of the function would walk its result for it.
    """
    return identity_refs.named_objects(func, _sentinel_like)


def _sentinel_like(value) -> bool:
    if type(value) is object:
        return True
    if isinstance(value, enum.Enum) or not compared_by_identity(value):
        return False
    return is_user_code_module(sys.modules.get(type(value).__module__))


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
    ``PurityChecks.surface_purity`` import stays local."""

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


#: What the observer calls an outbound connection.
_NETWORK_LABEL = observed_label(EffectKind.NETWORK)


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
        lines, first = getsourcelines(reader)
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


#: Past this, the per-argument hashes that NAME a changed argument are no
#: longer taken for the function (~7.4 ms for a 16 MB ndarray, so roughly a
#: 100 MB argument): the message then says "an argument". Whether the call
#: changed its arguments is still checked on every miss, whatever it costs --
#: a check retired for a big argument stored a library's in-place write, and
#: the warm run skipped it.
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
        tree = ast.parse(textwrap.dedent(getsource(fn)))
    except SOURCE_RETRIEVAL_ERRORS + (SyntaxError,):
        return False
    return name in unsafe_uses_of(tree, frozenset({name}), bare_args=False, mutating_methods_only=True)


class LearnedMutations:
    """Names a call was OBSERVED to mutate, per ``(code object, scope)``.

    Learned once, on the miss that saw it; from then on the closure and
    globals folds stop folding those names, which would otherwise move the
    key on every call.
    """

    def __init__(self) -> None:
        self._by_code: dict[tuple[Any, str], set[str]] = {}

    def of(self, code: Any, scope: str) -> frozenset[str] | set[str]:
        """The names learned for *code* in *scope* ("closure" or "global")."""
        return self._by_code.get((code, scope), frozenset())

    def learn(self, code: Any, scope: str, name: str) -> None:
        """Record that a call of *code* mutated *name* in *scope*."""
        self._by_code.setdefault((code, scope), set()).add(name)


#: Exact types whose instances cannot change once built (no subclass: one
#: can carry attributes).
_UNCHANGEABLE_TYPES = (str, int, float, bool, type(None), bytes, complex)


def _cannot_change(value: Any) -> bool:
    """Whether *value* can never change in place: an exact immutable
    primitive, or an exact tuple or frozenset of such values."""
    t = type(value)
    if t in _UNCHANGEABLE_TYPES:
        return True
    if t is tuple or t is frozenset:
        try:
            return all(map(_cannot_change, value))
        except RecursionError:
            return False
    return False


class PurityChecks:
    """What a cached function does besides returning its result: the static
    purity findings, the effects and argument mutations a first call is seen to
    make, and results that share state with the caller."""

    def __init__(
        self,
        config: CashConfig,
        registry: FunctionRegistry,
        args: ArgHasher,
        frozen: FrozenResults,
        reads: GlobalReads,
        values: GlobalValues,
        classes: ClassDataFold,
        mutations: LearnedMutations,
        notices: Notices,
    ) -> None:
        self._config = config
        self._registry = registry
        self._args = args
        self._frozen = frozen
        self._reads = reads
        self._values = values
        self._classes = classes
        self._mutations = mutations
        self._notices = notices
        # Functions the STATIC pass already reported on. The runtime effect
        # observer stays quiet for these: it would be a second warning about
        # the same function, and the user has already been told.
        self._static_flagged: set[str] = set()

    def warn_shared_result(self, func, func_name: str, result, args, kwargs) -> None:
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
        if self._registry.purity_mode(func_name) == "silent":
            return
        try:
            shared = self._shared_with(result, args, kwargs, func)
        except Exception:  # noqa: BLE001 - a diagnostic must never break a call
            return
        if shared is None:
            return
        what, name = shared
        self._notices.warn_once(
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
            # A sentinel the body returns is handed back as itself on a hit.
            if sentinel_ref(func, result) is not None:
                return None
            for name, value in list(globals_.items()):
                if value is result and is_mutable(value):
                    return "is the module global", name
        return None

    def learn_mutating_captures(self, func: Callable, func_name: str, watched: dict[str, tuple[str, str]]) -> None:
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
            value = None
            try:
                if scope == "closure":
                    cell = cells.get(name)
                    if cell is None:
                        continue
                    after = capture_digest(self._args, cell.cell_contents)
                elif scope == "carrier":
                    mapping, key = owner
                    if key not in mapping:
                        continue
                    value = mapping[key]
                    after = self._values.carried_global_hash(value, getattr(func, "__module__", None))
                elif scope == "binding":
                    after = self._values.carried_state_digest(resolve_binding(*owner))
                elif scope == "classdata":
                    after = self._classes.class_data_digest(*owner)[0]
                else:
                    # The mapping the BEFORE hash came from -- a helper's
                    # module, when this entry was folded on a helper's behalf.
                    g = owner if isinstance(owner, dict) else own_globals
                    if not isinstance(g, dict) or name not in g:
                        continue
                    value = g[name]
                    after = self._values.global_value_digest(value, plain_data_kind(value))
            except Exception:  # noqa: BLE001 - unhashable NOW; treat as unchanged
                continue
            if after == before:
                continue
            if scope in ("carrier", "global") and rng_carrier_kind(value) is not None:
                # A random generator the body drew from: a hit moves it on
                # to where the body left it (`replay_rng_state`), so its
                # state stays a true input and the next call keys on it.
                continue
            if scope in ("carrier", "binding", "instance", "classdata"):
                # What a callable carries, moved by calling it: a library's
                # own state (a generator advanced, a cache filled) or a memo
                # a callable instance keeps in `self`. Not a result input the
                # user rebinds: stop folding it.
                self._mutations.learn(code, "global", name)
                logger.debug("[CORE] %s: stopped keying what %s carries; calling it changes it", func_name, name)
                continue
            self._mutations.learn(code, scope, name)
            if scope == "closure":
                where = f"variable it captures '{name}'"
            else:
                module = owner.get("__name__") if isinstance(owner, dict) else None
                if module in MAIN_MODULE_NAMES and reader is not None:
                    module = resolve_main_module(reader)
                where = f"module global '{module}.{name}'" if module else f"module global '{name}'"
            site = describe_scope_use(reader, name, func)
            self._notices.warn_once(
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
                "`# @cash:assume-safe` on the line named, or "
                "`with cash.assume_safe():` around it.",
            )

    def refuses_identity_coupled(self, func_name: str, result: Any) -> bool:
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

        Checked here rather than inside ``ResultStore.store`` so the refusal
        lands beside ``cache_if``, BEFORE ``ResultStore.attach_lineage``: a value that is
        not stored must not carry a lineage hash pointing at an entry that was
        never written.

        KNOWN BOUNDARY: called at all four store sites (sync/async x
        non-iterator/single-chunk), which is every site where the value is in
        hand before anything is written.  A *multi*-chunk iterator is not
        covered -- ``ResultStore.stream_and_store`` has already written earlier chunks by the
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
        self._notices.warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            "",
            f"@cash.cache on {func_name}: result not cached. {reason}",
            code="CACHE-IDENTITY-COUPLED",
            fix="split the function: cache the part that computes the numbers, "
            "and draw the figure from them in an uncached function.",
        )
        return True

    def _checked_arguments(self, func_name: str, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """The arguments the in-place-change check looks at, canonicalised
        (`ArgHasher.normalize_call_args`): every one, except the parameters
        left out with ``ignore=`` or ``cash.Ignore[T]``.

        Ignoring a parameter is the user's statement that it does not matter
        to the call, as ``assume_safe=True`` is for an effect: hashing a
        200 MB scratch buffer twice on every miss to check it was the cost
        ``ignore=`` was meant to remove. ``key=`` still has every argument
        checked: it says how to tell calls apart, not that an argument does
        not matter.
        """
        cf = self._registry.cached.get(func_name)
        arg_key = getattr(cf, "arg_key", None)
        if arg_key is None or arg_key.key_fn is not None:
            return self._args.normalize_call_args(func_name, args, kwargs)
        normalized = self._args.normalize_call_args(func_name, args, kwargs, signature=cf.signature)
        return keyed_arguments(arg_key, cf.signature, func_name, args, kwargs, normalized)

    def checked_arguments_hash(self, func_name: str, args: tuple, kwargs: dict) -> str | None:
        """The hash the in-place-change check compares across the body: that
        of `PurityChecks._checked_arguments`, made as the key makes it
        (`ArgHasher.serialize_args`, or `ArgHasher.hash_payload` of what
        ``ignore=`` leaves), so that with ``ignore=`` the key's own argument
        hash is the one taken before the body. Raises what hashing raised."""
        cf = self._registry.cached.get(func_name)
        arg_key = getattr(cf, "arg_key", None)
        if arg_key is None or arg_key.key_fn is not None:
            return self._args.serialize_args(func_name, args, kwargs)
        return self._args.hash_payload(*self._checked_arguments(func_name, args, kwargs))

    def argument_snapshot(self, func_name: str, args: tuple, kwargs: dict) -> dict[str, str] | None:
        """``{parameter: hash}`` of the arguments that CAN change, before the body.

        An int, a str, a tuple of them: rebinding one inside the body (``n -=
        1``) is invisible to the caller, so they are left out, and most calls
        snapshot nothing. What remains lets `PurityChecks.check_argument_mutation` name the
        argument that moved. None when naming has been retired as too costly
        for this function (the check itself still runs), or nothing could be
        hashed.
        """
        cf = self._registry.cached.get(func_name)
        if cf is None or cf.argument_naming_retired:
            return None
        # The key was hashed a moment ago, on this thread: if that already cost
        # more than naming may, naming is retired before it pays -- a miss on
        # two million rows hashed them three times, once for the key, once
        # here and once after the body. Read from what
        # `ArgHasher.note_arg_cost` kept: it has already taken `ARG_COST.last`.
        cost = cf.arg_cost
        if cost is not None and cost[2] > MUTATION_CHECK_BUDGET_S:
            cf.argument_naming_retired = True
            return None
        started = _perf_counter()
        try:
            canon_args, canon_kwargs = self._checked_arguments(func_name, args, kwargs)
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
                snapshot[name] = self._args.hash_payload((value,), {})
            except Exception:  # noqa: BLE001 - unhashable: the whole-args check still runs
                continue
        if _perf_counter() - started > MUTATION_CHECK_BUDGET_S:
            cf.argument_naming_retired = True
        return snapshot

    def held_generators(self, func_name: str, args: tuple, kwargs: dict) -> list:
        """The random generators held as attributes by arguments whose key is
        not their content -- a ``__cash_key__`` or a registered hasher -- each
        with its state before the body (`capture_argument_carrier_states`).

        ``Sim(seed).run()`` keyed by ``__cash_key__`` returning the seed draws
        from ``self.rng``: the key does not move with the generator, and the
        in-place-change check hashes ``self`` by that same key, so the first
        draw was stored and served for every later call.
        """
        try:
            canon_args, canon_kwargs = self._checked_arguments(func_name, args, kwargs)
        except Exception:  # noqa: BLE001 - best effort, like the check itself
            canon_args, canon_kwargs = args, kwargs
        named = [(f"*args[{i}]", v) for i, v in enumerate(canon_args)] + list(canon_kwargs.items())
        held: list[tuple[str, Any]] = []
        for name, value in named:
            attrs = getattr(value, "__dict__", None)
            if not isinstance(attrs, dict) or not attrs:
                continue
            if cash_key_method_of_type(type(value)) is None and not self._args.keys_by_registration_only(value):
                continue
            held.extend((f"{name}.{attr}", v) for attr, v in attrs.items() if rng_carrier_kind(v) is not None)
        return capture_argument_carrier_states(held) if held else []

    def argument_identities(
        self, func_name: str, args: tuple, kwargs: dict
    ) -> tuple[dict[str, tuple[Any, list]], bool]:
        """``({parameter: (value, identity snapshot)}, covers_all)`` for the
        plain lists and tuples a call receives, before the body runs.

        The hash snapshot below is retired for a big argument and never covers
        a frozen one, which is exactly where ``rows.sort()`` on a million
        parsed rows, or a field rewritten in every row of a frozen result, got
        stored. Identities cost a fraction of a hash, so they are
        taken whatever the size (`_plain_data.identity_snapshot`).

        *covers_all* is True when every argument the check looks at is either
        such a list or tuple or a value that cannot change at all
        (`_cannot_change`). A plain list's leaves cannot change in place, so
        when no identity at any level moved, nothing in it did, and
        `PurityChecks.check_argument_mutation` needs no hash after the body:
        on a list of ten million ints that hash was a second of a 3.5 s miss.
        """
        try:
            canon_args, canon_kwargs = self._checked_arguments(func_name, args, kwargs)
        except Exception:  # noqa: BLE001 - best effort, like the check itself
            return {}, False
        found: dict[str, tuple[Any, list]] = {}
        covers_all = True
        named = [(f"*args[{i}]", v) for i, v in enumerate(canon_args)] + list(canon_kwargs.items())
        for name, value in named:
            if type(value) is list or type(value) is tuple:
                snapshot = _plain_data.identity_snapshot(value)
                if snapshot is None:
                    covers_all = False
                elif any(level is not None for level in snapshot):
                    found[name] = (value, snapshot)
            elif not _cannot_change(value):
                covers_all = False
        return found, covers_all

    def check_argument_mutation(
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
        not walk into. Such a mutation usually leaves an observable change in
        the arguments, so the cheapest way to find it is to look.

        The argument hash is already computed to build the cache key, so this
        re-runs exactly that and compares. `ArgHasher.serialize_args` canonicalises
        (kwargs order included) and is deterministic on unchanged input, which
        is what makes a difference mean *mutation* rather than noise.

        Runs on every miss, whatever the arguments' size: one more hash of
        them after the body. An argument that cannot be hashed again cannot
        be shown unchanged, so the result is not stored. A parameter left
        out with ``ignore=`` is not looked at (`PurityChecks._checked_arguments`).
        """
        if observer is None:
            return
        moved = moved_carriers(observer.held_generators) if observer.held_generators else []
        if moved:
            names = [where[1] for where, _pre, _post in moved]
            observer.mutated_args = names
            observer.record(
                "argument mutation",
                f"the call drew from the random generator {', '.join(repr(n) for n in names)}, which the "
                f"argument's __cash_key__ or registered hasher leaves out of the key -- the result was not "
                f"stored, so this call runs every time",
            )
            return
        if args_hash is None:
            return
        identities = observer.arg_identities
        if identities:
            moved = [
                name for name, (value, snapshot) in identities.items() if _plain_data.identity_changed(value, snapshot)
            ]
            if moved:
                for name in moved:
                    self._frozen.forget_container(identities[name][0])
                observer.mutated_args = moved
                observer.record(
                    "argument mutation",
                    f"the call changed {', '.join(repr(n) for n in moved)} in place -- "
                    f"the result was not stored, so this call runs every time",
                )
                return
        if observer.arg_identities_cover_all:
            # Every argument is plain data whose identities did not move, or
            # cannot change at all: nothing in them changed.
            return
        try:
            after = self.checked_arguments_hash(func_name, args, kwargs)
        except Exception:
            # Hashed for the key, not after the body: nothing says the body
            # left them as they were, and a hit would skip whatever it did.
            logger.debug("[PURITY] %s: arguments could not be hashed again", func_name, exc_info=True)
            observer.mutated_args = ["an argument"]
            observer.record(
                "argument mutation",
                "its arguments could not be hashed again after the call, so cash cannot tell "
                "whether it changed them -- the result was not stored, so this call runs every time",
            )
            return
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
        now = self.argument_snapshot(func_name, args, kwargs) or {}
        return [name for name, digest in before.items() if now.get(name) != digest]

    def make_effect_observer(self) -> EffectObserver:
        """An :class:`EffectObserver` scoped to this instance's cache dir.

        Excluding the cache directory is load-bearing: cash writes the entry
        it is computing, and without the exclusion every cached function would
        be observed writing a file and every one of them would warn.
        """

        cache_dir = getattr(self._config, "cache_dir", None)
        return EffectObserver(exclude_under=cache_dir)

    def report_observed_effects(self, func_name: str, observer: EffectObserver | None) -> None:
        """Warn once when the first call did something a hit will not do.

        Silent when:

        * ``assume_safe=True`` -- the user audited this function and said so;
          ``# @cash:assume-safe`` on a line that led to an effect waives that
          effect alone, and an effect performed while a ``with
          cash.assume_safe():`` block is open is waived too (see
          ``EffectObserver.record_effect``).
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
        if self._registry.purity_mode(func_name) == "silent":
            return
        covered: set[str] = set()
        if func_name in self._static_flagged:
            covered = static_effect_kinds(self._registry.purity_reports.get(func_name))
        effects = [(kind, detail) for kind, detail in observer.effects if kind not in covered]
        network = [detail for kind, detail in effects if kind == _NETWORK_LABEL]
        if network:
            effects = [(kind, detail) for kind, detail in effects if kind != _NETWORK_LABEL]
            self._report_observed_network(func_name, network)
        if not effects:
            return
        summary = "\n".join(dict.fromkeys(f"  {kind}: {detail}" for kind, detail in effects))
        self._notices.warn_once(
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
            "the way to it counts) or run it inside `with "
            "cash.assume_safe():`; @cash.cache(assume_safe=True) waives "
            "the whole function instead, including effects added later.",
        )

    def _report_observed_network(self, func_name: str, connections: list[str]) -> None:
        """A connection the static pass did not name: a network read.

        ``requests.Session().get``, ``from requests import get``, a client
        library -- the analyzer names only module-qualified calls, so these
        were seen only as a connection and filed as an effect a hit does not
        repeat. The hazard is the one KEY-NETWORK-READ names -- the answer is
        an input the key cannot see -- and so is the fix: ``ttl=`` silences
        it, and ``strict=True`` raises unless one is set. (A remote file a
        reader opened by URL is tracked and does not get here: its fetch is
        not observed, see ``reader_patches``.)
        """
        cf = self._registry.cached.get(func_name)
        if self._registry.effective_ttl(func_name, cf.ttl if cf is not None else None) is not None:
            return
        summary = "\n".join(dict.fromkeys(f"  {detail}" for detail in connections))
        if self._registry.purity_mode(func_name) == "strict":
            raise CashImpureFunctionError(
                f"@cash.cache(strict=True) on {func_name}: the first call "
                f"connected to a server, so its result depends on an answer "
                f"the cache key cannot see. Set ttl= to say how old a served "
                f"answer may be, or put `# @cash:assume-safe` on the line of "
                f"yours that led to it (or `with cash.assume_safe():` around "
                f"it).\n{summary}"
            )
        self._notices.warn_once(
            CashImpurityWarning,
            func_name,
            "observed_network_read",
            f"@cash.cache on {func_name}: the first call connected to a "
            f"server, so the result depends on what it returned, and that "
            f"answer is not part of the cache key. The first call's answer is "
            f"what every later call gets back -- in this process and in every "
            f"process after it -- until something changes the key.\n{summary}",
            code="KEY-NETWORK-READ",
            fix="bound how old a served answer may be with ttl= -- "
            "`@cash.cache(ttl=3600)` -- or pass what makes the answer new "
            "(a date, a version) as an argument, so it reaches the key. If "
            "the answer never changes, say so with `# @cash:assume-safe` on "
            "the line of yours that led to the connection, or `with "
            "cash.assume_safe():` around it.",
            once_per_version=True,
        )

    def _mutable_global_is_keyed(self, func_name: str, report: PurityReport, issue: Any) -> bool:
        """Is this ``mutable_global`` finding about a global the key already
        folds by value on every call?

        Then "cached results won't reflect changes to it" is false: a setter
        rebinding it, or a test patching it, makes the next call a new entry
        (a `configure()`-set module flag re-runs the function each time it
        changes). Kept for what the fold leaves out: callables, modules,
        classes (tracked their own way, or not at all), and a global the
        function itself writes.
        """

        name = getattr(issue, "subject", "")
        if getattr(issue, "kind", None) != ISSUE_MUTABLE_GLOBAL or not name:
            return False
        func = self._registry.functions.get(func_name)
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
            if name not in self._reads.read_global_data_names(reader):
                return False
        except Exception:  # noqa: BLE001 - user source; a heuristic must not break a call
            return False
        value = module_ns[name]
        if isinstance(value, types.ModuleType):
            # `conf.RATE` reads of a module of the user's are folded by value
            # (`ModuleAttrFold.module_attr_parts`); "mutated elsewhere" is `conf.RATE = ...`.
            return is_user_module(value, own_package(reader))
        if isinstance(value, type):
            return False
        return not (callable(value) and not isinstance(value, (dict, list, tuple, set)))

    def surface_purity(
        self,
        func_name: str,
        report: PurityReport,
        mode: str,
        *,
        per_report: bool = False,
    ) -> None:
        """Turn a `PurityReport` into warnings or an exception.

        Called once per function on first call (after first
        ``CallRunner._analyze_dependencies``).

        * ``warn`` (default): one-shot `CashImpurityWarning`
          summarising issues; also recorded in
          ``cache_info()['warnings']``.
        * ``silent`` (``assume_safe=True``): the user has audited
          this; suppress the warning. The report is still stored
          so helper source hashes invalidate correctly.
        * ``strict``: raise `CashImpureFunctionError`. Opaque
          callees count as issues in this mode (paranoid).

        ``per_report``: *func_name* is a closure, which shares its name with
        every other closure its factory makes but not its helpers. A warning
        is then shown once per distinct finding rather than once per name, so
        a second closure reaching an impure helper is told.
        """
        issues = self._reported_issues(func_name, report, mode)
        if not issues or mode == "silent":
            return
        if mode != "strict":
            # strict=True keeps every issue in the one exception it raises:
            # there, every issue is a hard stop and splitting the report
            # would hide half of it.
            _raise_untrackable(func_name, issues)
            issues = self._warn_ambient_reads(func_name, issues, per_report)
            issues = self._warn_network_reads(func_name, issues, per_report)
        if not issues:
            return
        self._static_flagged.add(func_name)
        summary = format_issues_summary(issues)
        if mode == "strict":
            raise CashImpureFunctionError(
                f"@cash.cache(strict=True) on {func_name}: purity issues "
                f"detected. Fix the function, mark callees with "
                f"@pure / @stateful, put `# @cash:assume-safe` on the lines you "
                f"have audited (or `with cash.assume_safe():` around them), or "
                f"relax to assume_safe=True.\n{summary}"
            )
        self._warn_side_effects(func_name, summary, per_report)

    def _reported_issues(self, func_name: str, report: PurityReport, mode: str) -> list:
        """The issues of *report* to surface under *mode*: without those the
        key already covers, without network reads a ``ttl=`` answers, and
        with the opaque callees under ``strict``."""
        issues = [i for i in report.issues if not self._mutable_global_is_keyed(func_name, report, i)]
        if any(getattr(i, "kind", None) == ISSUE_NETWORK_READ for i in issues):
            # Named statically, so the observer does not report the same read
            # as a connection -- whether or not `_warn_network_reads` warns.
            self._static_flagged.add(func_name)
            cf = self._registry.cached.get(func_name)
            if self._registry.effective_ttl(func_name, cf.ttl if cf is not None else None) is not None:
                # `ttl=` is the answer to "how old may a fetched answer be":
                # once one is set, the question has been answered.
                issues = [i for i in issues if getattr(i, "kind", None) != ISSUE_NETWORK_READ]
        if mode == "strict" and report.opaque_callees:
            opaque_list = ", ".join(report.opaque_callees[:5])
            if len(report.opaque_callees) > 5:
                opaque_list += f", ... +{len(report.opaque_callees) - 5} more"
            issues.append(make_opaque_issue(func_name, opaque_list))
        return self._ignored_mutations_unchecked(func_name, issues)

    def _ignored_mutations_unchecked(self, func_name: str, issues: list) -> list:
        """*issues*, with a finding that the function changes an ignored
        parameter in place saying what cash does about it: nothing
        (`PurityChecks._checked_arguments`). The analyzer's wording -- "a call
        that makes it is not stored" -- holds only for an argument the
        in-place-change check looks at."""
        cf = self._registry.cached.get(func_name)
        arg_key = getattr(cf, "arg_key", None)
        if arg_key is None or arg_key.key_fn is not None:
            return issues
        own = {func_name, getattr(cf.func, "__qualname__", None)}
        rewritten = []
        for issue in issues:
            text = getattr(issue, "description", "")
            if getattr(issue, "where", None) in own and ARGUMENT_MUTATION_NOT_STORED in text:
                for name in arg_key.ignored:
                    if f"the argument '{name}' in place" in text:
                        text = text.replace(
                            ARGUMENT_MUTATION_NOT_STORED,
                            f"'{name}' is ignored, so cash does not check it: the call is stored, and a "
                            f"cache hit does not make that change",
                        )
                        issue = dataclasses.replace(issue, description=text)
                        break
            rewritten.append(issue)
        return rewritten

    def _warn_ambient_reads(self, func_name: str, issues: list, per_report: bool) -> list:
        """Warn about the ambient reads among *issues*; return the rest.

        Ambient reads get their own warning, not the side-effects one. The
        hazard is the opposite shape -- nothing is skipped, a hidden INPUT is
        frozen -- and so is the fix: pass the value in as an argument, where
        it reaches the key. Filing them under "likely side effects" told the
        user to audit for writes that are not there, and left the actual
        failure (a nightly job whose `date.today()` is the night it first
        ran) unnamed.
        """
        ambient = [i for i in issues if getattr(i, "kind", None) == ISSUE_AMBIENT_READ]
        if ambient:
            issues = [i for i in issues if getattr(i, "kind", None) != ISSUE_AMBIENT_READ]
            self._static_flagged.add(func_name)
            self._notices.warn_once(
                CashImpurityWarning,
                func_name,
                _warning_slot("ambient", format_issues_summary(ambient), per_report),
                f"@cash.cache on {func_name}: the body reads ambient state "
                f"(the clock, the environment, a fresh UUID). That value is "
                f"not part of the cache key, so the first call's answer is "
                f"what every later call gets back -- in this process and in "
                f"every process after it.\n{format_issues_summary(ambient)}",
                code="KEY-AMBIENT-READ",
                fix="pass the value in as an argument -- `f(now=datetime.now())` "
                "-- so it reaches the cache key and a new value means a new "
                "entry. If freezing it is what you want, say so with "
                "`# @cash:assume-safe` on that line, or `with "
                "cash.assume_safe():` around it.",
                once_per_version=True,
            )
        return issues

    def _warn_network_reads(self, func_name: str, issues: list, per_report: bool) -> list:
        """Warn about the network and database reads among *issues*; return the rest.

        Its own advisory, for the same reason as an ambient read: nothing is
        skipped, an input the key cannot see is frozen. Unlike the clock it
        has a knob made for it, `ttl=`, which silences it
        (`_reported_issues`). Under strict=True it raises with the other
        issues unless a ttl= is set.
        """
        remote = [i for i in issues if getattr(i, "kind", None) == ISSUE_NETWORK_READ]
        if remote:
            issues = [i for i in issues if getattr(i, "kind", None) != ISSUE_NETWORK_READ]
            self._notices.warn_once(
                CashImpurityWarning,
                func_name,
                _warning_slot("network_read", format_issues_summary(remote), per_report),
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
                "on that line, or `with cash.assume_safe():` around it.",
                once_per_version=True,
            )
        return issues

    def _warn_side_effects(self, func_name: str, summary: str, per_report: bool) -> None:
        """The ``warn``-mode warning for the side effects in *summary*."""
        self._notices.warn_once(
            CashImpurityWarning,
            func_name,
            _warning_slot("purity", summary, per_report),
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
                "you have audited, or `with cash.assume_safe():` around "
                "several, or refactor; @cash.cache(assume_safe=True) waives "
                "the whole function instead, including anything added to it "
                "later. No waiver changes the function's cache key."
            ),
            once_per_version=True,
        )


def _warning_slot(kind: str, findings: str, per_report: bool) -> str:
    """The dedup slot of a purity warning of *kind*: one per function, or with
    *per_report* one per distinct *findings* (`PurityChecks.surface_purity`)."""
    if not per_report:
        return kind
    return f"{kind}:{hashlib.sha256(findings.encode('utf-8')).hexdigest()[:16]}"


def _raise_untrackable(func_name: str, issues: list) -> None:
    """Raise when *issues* include an untrackable dependency.

    Untrackable-dependency patterns (eval/exec/compile, getattr(obj,name)()
    dynamic dispatch, importlib.import_module) RAISE by default, even in
    the ordinary "warn" mode: cash cannot see an edit to a dependency it
    resolves from a runtime value, so a cached result can go silently
    stale, and caching correctness can no longer be guaranteed. The user
    must acknowledge the risk with assume_safe=True (the ``silent`` mode,
    which never reaches here) to cache anyway.
    """
    untrackable = [i for i in issues if getattr(i, "kind", None) == ISSUE_UNTRACKABLE_DEP]
    if untrackable:
        untrackable_summary = format_issues_summary(untrackable)
        raise CashImpureFunctionError(
            f"@cash.cache on {func_name}: a dependency is resolved from a "
            f"runtime value, so cash cannot tell when it changes and a cached "
            f"result could be silently stale. Caching correctness cannot be "
            f"guaranteed for this function.\nPut `# @cash:assume-safe` on "
            f"the line named below (or `with cash.assume_safe():` around "
            f"it) to accept the risk for that statement alone, pass "
            f"@cash.cache(assume_safe=True) to waive the whole function, or "
            f"refactor to a statically-named call.\n{untrackable_summary}"
        )

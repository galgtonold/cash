"""Folding what a function captures -- closure cells, parameter defaults, the
instance it is bound to, its helpers' captures -- into its key."""

from __future__ import annotations

import ast
import dis
import hashlib
import inspect
import pickle
import textwrap
import types
import weakref
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from ..effect_observer import line_waived
from ..exceptions import SOURCE_RETRIEVAL_ERRORS, CashCacheIneffectiveWarning
from ..purity_analyzer import REPORTED_METHODS
from ..value_types import IMMUTABLE_VALUE_TYPES
from .arg_hashing import CODE_VALUE_TYPES, is_opaque
from .call_state import CAPTURE_WATCH, KeyBuildFailed
from .code_identity import code_fingerprint, hash_callable_source, is_user_code_object

if TYPE_CHECKING:
    from .arg_hashing import ArgHasher
    from .globals_fold import GlobalsFold
    from .purity_checks import LearnedMutations
    from .reporting import Notices


def is_immutable_capture(v: Any, _depth: int = 0) -> bool:
    """True for values that are immutable and so define a closure's
    behaviour without drifting between calls. Mutable captures (dict/list/
    set/objects) are excluded: they are typically side-effect accumulators
    (e.g. a hit counter) whose value changes every call - folding those into
    the key would make every call miss."""
    if _depth > 8:
        return False
    if isinstance(v, (bool, int, float, complex, str, bytes, type(None))):
        return True
    if isinstance(v, (tuple, frozenset)):
        return all(is_immutable_capture(x, _depth + 1) for x in v)
    return False


def unsafe_uses_of(
    tree: ast.AST,
    names: set[str],
    *,
    bare_args: bool = True,
    mutating_methods_only: bool = False,
    waived: Callable[[ast.AST], bool] | None = None,
) -> frozenset:
    """Return the subset of *names* the AST body *may mutate*.

    Disqualifying uses of a name ``n``: method calls on it
    (``n.append(...)``), passing it as a bare argument (the callee may
    mutate), subscript/attribute stores or aug-assigns rooted at it, and
    ``del``. Iteration, subscript reads, and arithmetic stay safe. Shared by
    the closure-capture and module-global folds.

    Two knobs, and both exist to move a *suspicion* out of the HARD set and
    into the provisional one, where it is folded and then confirmed at
    runtime by ``PurityChecks.learn_mutating_captures``.

    ``bare_args=False`` drops the "passed as an argument" rule. That refusal
    was over-broad and chose the worse failure: `sum(G)`, `len(G)`,
    `helper(G)` and `model.predict(G)` all put `G` beyond it, so a later
    `G = ...` never reached the key and the function served a stale value
    for ever, silently.

    ``mutating_methods_only=True`` narrows the method-call rule to methods
    that actually write -- ``append``, ``update``, ``sort`` and their
    relatives, the same table the purity analyzer uses. "Any method, since
    we cannot prove purity" made the same over-broad choice one level down,
    and it cost: a lookup table read as
    ``ALIASES.get(v, v)`` never reached the key, so editing the table
    published stale labels with nothing to see. `ALIASES[v]`, `v in
    ALIASES`, `d = ALIASES; d.get(v)` and a bare read all tracked
    correctly, which is what made it so hard to believe.

    ``waived`` skips uses on a ``# @cash:assume-safe`` line: the effect
    there was audited as one a hit may lose (see `_waived_use_filter`).
    """
    unsafe: set[str] = set()
    write_methods: frozenset[str] = frozenset()
    if mutating_methods_only:
        write_methods = REPORTED_METHODS
    for node in ast.walk(tree):
        if waived is not None and isinstance(node, (ast.Call, ast.stmt)) and waived(node):
            continue
        if isinstance(node, ast.Call):
            f = node.func
            if (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id in names
                and (not mutating_methods_only or f.attr in write_methods)
            ):
                unsafe.add(f.value.id)
            if bare_args:
                for a in list(node.args) + [kw.value for kw in node.keywords]:
                    if isinstance(a, ast.Starred):
                        a = a.value
                    if isinstance(a, ast.Name) and a.id in names:
                        unsafe.add(a.id)
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]
            else:
                targets = node.targets
            for t in targets:
                root = t
                while isinstance(root, (ast.Subscript, ast.Attribute, ast.Starred)):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in names:
                    unsafe.add(root.id)
    return frozenset(unsafe)


def waived_use_filter(func: Callable) -> Callable[[ast.AST], bool] | None:
    """A predicate: is a node of *func*'s dedented source on a waived line?

    Line numbers in that tree count from the ``def``; the waiver is read
    from the file. ``None`` when the file position is unknown.
    """
    try:
        filename = inspect.getsourcefile(func) or inspect.getfile(func)
        first = inspect.getsourcelines(func)[1]
    except SOURCE_RETRIEVAL_ERRORS:
        return None
    if not filename:
        return None

    offset = max(first, 1) - 1

    def waived(node: ast.AST) -> bool:
        start = getattr(node, "lineno", None)
        if start is None:
            return False
        end = getattr(node, "end_lineno", None) or start
        return any(line_waived(filename, offset + n) for n in range(start, end + 1))

    return waived


def defaults_of(func: Callable) -> tuple[tuple, dict]:
    """The parameter defaults that decide what *func* computes.

    ``__defaults__`` (positional/keyword params) and ``__kwdefaults__``
    (keyword-only params) are separate containers; both are collected.

    Wrapped callees are walked too. ``func.__defaults__`` is what the call
    literally binds, but when *func* is a ``functools.wraps`` wrapper its own
    defaults are typically empty (a ``*args, **kwargs`` passthrough) while the
    values that actually decide the result sit on ``__wrapped__`` — which is
    also what ``inspect.signature`` reports and therefore what
    ``ArgHasher.normalize_call_args`` binds. Folding every level is the conservative
    choice: folding a default that turns out not to bind costs at most a
    one-time miss, whereas missing one that does bind is a silent wrong
    answer.
    """
    pos: list[Any] = []
    kwd: dict[str, Any] = {}
    seen: set[int] = set()
    fn: Any = func
    depth = 0
    while fn is not None and id(fn) not in seen and depth < 8:
        seen.add(id(fn))
        pos.extend(getattr(fn, "__defaults__", None) or ())
        # Qualify by depth so a wrapper and its wrappee can't collide on a
        # shared kwonly name; sort so dict order never leaks into the key.
        level_kwd = getattr(fn, "__kwdefaults__", None) or {}
        for name in sorted(level_kwd):
            kwd[f"{depth}:{name}"] = level_kwd[name]
        fn = getattr(fn, "__wrapped__", None)
        depth += 1
    return tuple(pos), kwd


def fingerprint_default(v: Any) -> Any:
    """Replace a plain function/method default with a digest of its source.

    Restricted to functions, methods and builtins: their behaviour IS their
    code. An arbitrary callable INSTANCE is left alone so it takes the
    unhashable path rather than being keyed on its class and silently
    sharing entries across instances with different state.
    """
    if inspect.isfunction(v) or inspect.ismethod(v) or inspect.isbuiltin(v):
        return f"__cash_callable__:{hash_callable_source(v)}"
    return v


def iter_code_scopes(code: types.CodeType) -> Iterator[types.CodeType]:
    """Yield *code* and every code object nested inside it, recursively.

    A generator expression, comprehension, or ``lambda`` compiles to its
    OWN code object hung off the enclosing ``co_consts``, so anything it
    references is invisible in the outer ``co_names`` / instruction stream.
    Walking the const tree is the same trick the
    bytecode hash uses (``tracking/function_tracker.py``
    ``_update_code_object_hash``) for exactly this reason.

    Comprehensions nest, so this recurses. Note that CPython 3.12+ inlines
    list/set/dict comprehensions into the enclosing scope (PEP 709) — those
    already land in the outer ``co_names``; generator expressions and
    lambdas still get their own scope on every version.
    """
    yield code
    for const in code.co_consts or ():
        if isinstance(const, types.CodeType):
            yield from iter_code_scopes(const)


class CaptureAnalysis:
    """Which of a closure's captured variables the function body may change,
    read from its code and source once per code object (closures from one
    factory share it)."""

    def __init__(self) -> None:
        # code object -> frozenset of reassigned freevars
        self._deref_writes: dict = {}
        # code object -> frozenset of free vars with capture-unsafe uses
        self._use_cache: dict = {}
        # code object -> closure free vars folded only provisionally: passed to
        # a call, so folded and then confirmed by observation. Kept in
        # lockstep with the cache above.
        self._provisional: dict = {}

    def written_freevars(self, code: Any) -> frozenset:
        """Free-variable names the function reassigns (``STORE_DEREF`` /
        ``DELETE_DEREF``) - i.e. ``nonlocal`` counters that drift between calls.
        Cached per code object (closures share a code object per factory)."""
        cache = self._deref_writes
        hit = cache.get(code)
        if hit is not None:
            return hit

        written = frozenset(
            instr.argval for instr in dis.get_instructions(code) if instr.opname in ("STORE_DEREF", "DELETE_DEREF")
        )
        if len(cache) < 4096:
            cache[code] = written
        return written

    def unsafe_uses(self, func: Callable) -> frozenset:
        """Free-variable names whose captured object *may be mutated* by the
        function body.

        A capture is only content-foldable into the cache key when the body
        provably just READS it. Disqualifying uses of a free variable ``n``:
        method calls on it (``n.append(...)`` — any method, since we can't
        prove purity), passing it as a bare argument (the callee may mutate),
        subscript/attribute stores or aug-assigns rooted at it, and ``del``.
        Iteration, subscript reads, and arithmetic stay safe.

        Cached per code object (closures from one factory share it). When the
        source is unavailable, every free var is reported unsafe.
        """
        code = getattr(func, "__code__", None)
        if code is None:
            return frozenset()
        cached = self._use_cache.get(code)
        if cached is not None:
            return cached
        freevars = set(code.co_freevars or ())
        provisional: frozenset = frozenset()
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        except SOURCE_RETRIEVAL_ERRORS:
            # No source: the conservative answer. Nothing can be told
            # apart, so nothing is folded and nothing is provisional.
            result = frozenset(freevars)
        else:
            # Only mutations visible in this function's own source disqualify a
            # capture outright. "Passed to a call" is provisional: folded, then
            # confirmed by observation, so `sum(data)` does not put `data`
            # beyond the fold, where the closure would serve stale forever.
            result = unsafe_uses_of(
                tree,
                freevars,
                bare_args=False,
                mutating_methods_only=True,
            )
            suspected = unsafe_uses_of(tree, freevars) - result
            provisional = unsafe_uses_of(tree, suspected, waived=waived_use_filter(func))
            # Only on waived lines: as for globals (`GlobalsFold.read_global_data_names`).
            result = result | (suspected - provisional)
        if len(self._use_cache) < 4096:
            self._use_cache[code] = result
            # Kept in lockstep with the cache above so the two can never
            # disagree about a code object.
            self._provisional[code] = provisional
        return result

    def provisional(self, code: Any) -> frozenset | None:
        """The free vars of *code* folded only provisionally, or None if unknown."""
        return self._provisional.get(code)


class HelperIdentity:
    """A helper's identity for the key: its code, its parameter defaults and
    the immutable values its closure captured."""

    def __init__(self, args: ArgHasher, captures: CaptureAnalysis) -> None:
        self._args = args
        self._captures = captures
        # id(helper) -> (helper, __defaults__, __kwdefaults__, identity); see
        # `identity`. Holding the helper keeps its id from being recycled while
        # the entry lives.
        self._defaults_memo: dict[int, tuple[Any, Any, Any, str]] = {}

    def _capture_part(self, fn: Callable) -> str:
        """Digest of the IMMUTABLE values a helper's closure captured, or "".

        A decorator's arguments live there: ``@scale(10)`` builds a wrapper
        whose closure holds ``k=10``, so ``@scale(100)`` -- or ``@scale(K)``
        after ``K`` changed -- ran different code under an identical source and
        was served stale. Immutable values only, and never a variable the
        function reassigns (``nonlocal calls; calls += 1``): decorators often
        keep caches, counters and registries in their closures, and folding
        state that drifts on every call would make every call miss. Captured
        FUNCTIONS are followed as helpers in their own right, not here.
        """
        closure = getattr(fn, "__closure__", None)
        code = getattr(fn, "__code__", None)
        if not closure or code is None:
            return ""
        written = self._captures.written_freevars(code)
        unsafe: frozenset | None = None
        captures = []
        for name, cell in zip(code.co_freevars, closure):
            if name in written:
                continue
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if callable(value) or isinstance(value, types.ModuleType):
                continue
            if not (is_immutable_capture(value) or isinstance(value, IMMUTABLE_VALUE_TYPES)):
                # A container the helper only READS is data like any other:
                # `lambda: when` with `when` a list, a dict -- or a datetime
                # before the type list above had it -- gave every value ONE
                # entry, so the standard frozen-clock fixture served July's
                # answer to a March test. What the body mutates
                # (a decorator's cache dict, a counter list) stays out, as
                # before: folding it would make every call miss.
                if unsafe is None:
                    unsafe = self._captures.unsafe_uses(fn)
                if name in unsafe or not isinstance(value, (list, dict, set, tuple, frozenset)):
                    continue
            captures.append((name, value))
        if not captures:
            return ""
        try:
            return self._args.hash_payload(tuple(captures), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError):
            return ""

    def identity(self, fn: Callable) -> str:
        """A helper's identity for the key: its code, AND its parameter defaults.

        A default is evaluated once, at ``def`` time, and lives on the function
        object -- so ``def shrink(v, alpha=ALPHA)`` reads the same after
        ``ALPHA`` changes, the source digest does not move, and global folding
        never sees the name (it is not read in the body): a service whose ridge
        penalty was a helper's default served 8 wrong answers in 8. The cached function's own defaults were already
        folded (``ClosureFold.fold_defaults``); now every followed helper's are, by value,
        through the same payload hasher and the same callable fallback.

        Not inside ``_hash_callable_source``'s memo: that is keyed per CODE
        object, and two closures from one factory share a code object while
        holding different defaults.
        """
        source = hash_callable_source(fn)
        if isinstance(fn, type):
            # A class's own source says nothing about what it inherits, and
            # this channel is what the key folds: ``Worker(Base)`` calling an
            # inherited ``run`` kept serving the old answer after ``Base.run``
            # was rewritten -- 20 where an uncached run gives 500, in one file.
            # An OPAQUE base still contributes nothing, as for a class passed as an argument.
            bases = [
                hash_callable_source(base)
                for base in fn.__mro__[1:]
                if base is not object and not is_opaque(base) and is_user_code_object(base)
            ]
            if bases:
                source = f"{source}:bases:{','.join(bases)}"
        captured = self._capture_part(fn)
        if captured:
            source = f"{source}:captures:{captured}"
        defaults = getattr(fn, "__defaults__", None)
        kwdefaults = getattr(fn, "__kwdefaults__", None)
        memo_key = id(fn)
        cached = self._defaults_memo.get(memo_key)
        if cached is not None and cached[0] is fn and cached[1] is defaults and cached[2] is kwdefaults:
            return cached[3]
        pos, kwd = defaults_of(fn)
        try:
            digest = self._args.hash_payload(pos, kwd)
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError):
            try:
                digest = self._args.hash_payload(
                    tuple(fingerprint_default(v) for v in pos),
                    {k: fingerprint_default(v) for k, v in kwd.items()},
                )
            except (TypeError, pickle.PicklingError, AttributeError, OverflowError) as e:
                bad_type = self._args.first_unhashable_arg_type(pos, kwd)
                name = getattr(fn, "__qualname__", repr(fn))
                raise KeyBuildFailed(
                    "KEY-UNHASHABLE-DEFAULT",
                    f"@cash.cache: a parameter default of type {bad_type} on the helper "
                    f"{name} could not be hashed ({type(e).__name__}), so the call ran "
                    f"uncached rather than risk serving a result computed under a "
                    f"default that changed.",
                    f"get the value out of {name}'s signature -- build it in the body or "
                    f"pass it at the call site -- or register a hasher with "
                    f"cash.register_hasher({bad_type}, ...).",
                ) from e
        identity = f"{source}:defaults:{digest}"
        if len(self._defaults_memo) >= 4096:
            self._defaults_memo.clear()
        self._defaults_memo[memo_key] = (fn, defaults, kwdefaults, identity)
        return identity

    def fingerprint_default(self, v: Any) -> Any:
        """`_fingerprint_default`, plus what a FUNCTION default carries.

        A factory-built callable as a default (`def run(xs, fn=make(3))`)
        shares its source with every other one the factory makes; the value it
        was built with lives in its closure, and was not keyed -- `make(3)` ->
        `make(1)` served the old result. `HelperIdentity.identity`
        adds its immutable captures and its own defaults.
        """
        if inspect.isfunction(v):
            return f"__cash_callable__:{self.identity(v)}"
        return fingerprint_default(v)


class ClosureFold:
    """The closure, default and bound-instance folds of the state segment."""

    def __init__(
        self,
        args: ArgHasher,
        captures: CaptureAnalysis,
        helpers: HelperIdentity,
        globals_fold: GlobalsFold,
        mutations: LearnedMutations,
        notices: Notices,
    ) -> None:
        self._args = args
        self._captures = captures
        self._helpers = helpers
        self._globals = globals_fold
        self._mutations = mutations
        self._notices = notices
        # function object -> digest of its parameter defaults, for defaults that
        # are immutable and therefore cannot drift between calls.
        # Weak so the memo dies with the function instead of pinning it (and so
        # a later function object can never inherit a dead one's entry by
        # id-reuse). Mutable defaults are deliberately absent: they must be
        # re-hashed per call to stay correct.
        self._defaults_pins: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def fold_closure(self, func: Callable, func_name: str, state_hash: str, _depth: int = 0) -> str:
        """Mix a fingerprint of *func*'s captured free variables into the
        state hash.

        Two closures produced by the same factory share a source AND a qualname
        (``factory.<locals>.f``) but capture different values - so without this
        they collide on the same cache key and return each other's results (a
        silent wrong answer, e.g. ``make(2)`` vs ``make(5)``).

        Immutable captures are folded by value. Mutable captures (lists,
        dicts, arrays — e.g. a weights vector) are folded by CONTENT HASH,
        but only when the body provably just reads them: captures
        the function reassigns (``nonlocal`` counters) or may mutate in place
        (accumulators) are skipped, so their keys don't drift call-to-call.
        A side effect of per-call content hashing: externally mutating a
        folded capture correctly invalidates the closure's entries.
        """
        closure = getattr(func, "__closure__", None)
        code = getattr(func, "__code__", None)
        if not closure or code is None:
            return state_hash
        freevars = getattr(code, "co_freevars", ()) or ()
        # Free vars the function REASSIGNS (nonlocal counters) drift across calls
        # even when their value type is immutable - exclude them.
        written = self._captures.written_freevars(code)
        unsafe = self._captures.unsafe_uses(func)
        # Captures excluded ONLY because they were passed to a call. Folded, then
        # confirmed at runtime -- same treatment as module globals. A
        # missing entry means "unknown": watch it rather than fold it blind.
        provisional = self._captures.provisional(code)
        learned_mutating = self._mutations.of(code, "closure")
        captures = []
        for name, cell in zip(freevars, closure):
            if name in written or name in learned_mutating:
                continue
            try:
                v = cell.cell_contents
            except ValueError:
                continue
            if getattr(v, "_cash_cached", False):
                # A captured CACHED function is what it computes: its
                # dependency state, as a registry holding one counts it
                # (`GlobalsFold.data_callable_identity`). Not cash's wrapper around
                # it, whose closure holds this Cash instance and the
                # function's spec: those were content-hashed into the key on
                # every call, backend and all, while the write thread changed
                # the backend's dicts -- "dictionary changed size during
                # iteration", and the call ran uncached.
                captures.append((name, self._globals.data_callable_identity(v)))
                continue
            # A captured FUNCTION is its code, so fold its source. Reaching
            # this before the `unsafe` check is the point: a capture the body
            # PASSES TO A CALL is marked unsafe and skipped (watch it, don't
            # fold it blind), and calling is exactly what you do
            # with a captured function. So the strategy-factory shape
            #
            #     def make(weight_fn):
            #         @cash.cache
            #         def score(px, n): return f(px, weight_fn(n))
            #         return score
            #
            # folded NOTHING: two scorers built with different weightings share
            # a source and a qualname (`make.<locals>.score`), collided on one
            # key, and returned each other's results. Measured: `flat` and
            # `ramp` both returning 0.025001250062501867, one body execution.
            #
            # A call cannot mutate a function, so the reason `unsafe` exists
            # does not apply. Same predicate as `_fingerprint_default`, and the
            # same deliberate limit: functions, methods and builtins only. An
            # arbitrary callable INSTANCE takes the paths below rather than being
            # keyed on its class and silently sharing entries across instances
            # holding different state.
            fingerprint = fingerprint_default(v)
            if fingerprint is not v:
                # Source text alone collides for two lambdas sharing a line
                # (`a(lambda: "AAA"), a(lambda: "BBB")` is ONE line, so
                # `inspect.getsource` returns the same string for both).
                # Measured: both arms returned "AAA". Their code objects differ.
                inner_code = getattr(v, "__code__", None)
                if inner_code is not None:
                    fingerprint = f"{fingerprint}:{code_fingerprint(inner_code)}"
                # Source alone is not enough: a factory-built helper has the
                # SAME source for every parameter it was built with, so
                # `outer(2)` and `outer(3)` fingerprint identically and collide
                # again one level down (measured: both returned 20). Recurse so
                # the captured function's own captures fold under the same
                # rules. Bounded, because a wrong answer is worth a few frames
                # and a cycle is not.
                if _depth < 4:
                    fingerprint = self.fold_closure(
                        v,
                        f"{func_name}.{name}",
                        str(fingerprint),
                        _depth + 1,
                    )
                captures.append((name, fingerprint))
                continue

            if is_immutable_capture(v):
                captures.append((name, v))
            elif name not in unsafe:
                # Read-only mutable capture: fold its content hash.
                # Unhashable content is skipped.
                try:
                    h = self._args.hash_payload((v,), {})
                except (TypeError, pickle.PicklingError, AttributeError, OverflowError):
                    continue
                captures.append((name, h))
                pending = CAPTURE_WATCH.get()
                if pending is not None and (provisional is None or name in provisional):
                    pending[name] = (h, "closure", None, func)
        if not captures:
            return state_hash
        clo = self._args.serialize_args(func_name, tuple(captures), {})
        if not clo:
            return state_hash
        return hashlib.sha256(f"{state_hash}:closure:{clo}".encode()).hexdigest()

    def fold_defaults(
        self,
        func: Callable,
        func_name: str,
        state_hash: str,
    ) -> str | None:
        """Mix the callee's parameter defaults into the state hash.

        A default is an input to the result exactly like a passed argument, but
        it lives on the FUNCTION OBJECT, not in the code object — so the bytecode
        fingerprint the state hash falls back to when source is unavailable
        (functions defined in an IPython cell, the documented ML path) cannot see
        it. Editing ``n_estimators=300`` to ``400`` left the key byte-identical
        and returned the 300-tree model on an instant HIT while
        ``inspect.signature`` reported 400 — a wrong answer that reads as a
        finding ("accuracy has plateaued") rather than as a bug.

        Defaults are hashed by VALUE through the same payload hasher arguments
        use, so ``register_hasher`` and the pandas/numpy-aware hashers apply
        identically. Returns ``None`` when a default cannot be hashed; the caller
        must then refuse to cache, because silently ignoring it would resurrect
        exactly the silent staleness this fold exists to prevent.
        """
        # Memo first: this runs on EVERY decorated call, so the hot path must be
        # one lookup plus one hash, with no re-walk of the function.
        # Not every callable can be weak-referenced (numpy's dispatcher can't),
        # and WeakKeyDictionary raises on LOOKUP too, not just on store.
        try:
            entry = self._defaults_pins.get(func)
            pinnable = True
        except TypeError:
            entry = None
            pinnable = False
        if entry is not None:
            # Validate against the live containers rather than trusting the
            # function object's identity: `f.__defaults__ = (400,)` rebinds them
            # on the SAME object, and a memo keyed on identity alone would pin
            # the old digest and hand back a stale result — the very failure
            # this fold exists to prevent. Only immutable defaults are pinned,
            # so comparing the containers by value is sound (and is a cheap
            # C-level compare of a tiny tuple/dict).
            pin_pos, pin_kwd, digest = entry
            if pin_pos == getattr(func, "__defaults__", None) and pin_kwd == (
                getattr(func, "__kwdefaults__", None) or {}
            ):
                return hashlib.sha256(f"{state_hash}:defaults:{digest}".encode("utf-8")).hexdigest()
        pos, kwd = defaults_of(func)
        try:
            digest = self._args.hash_payload(pos, kwd)
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError):
            # A callback default (`def f(x, key=lambda v: v)`) is unpicklable but
            # is not opaque: its SOURCE defines it, which is the same fingerprint
            # register_hasher embeds for a hasher. Retry with function-valued
            # defaults replaced by that -- strictly better than dropping them (an
            # edited lambda now invalidates) and it keeps such functions
            # cacheable, which a bare refuse-to-cache would not.
            try:
                digest = self._args.hash_payload(
                    tuple(self._helpers.fingerprint_default(v) for v in pos),
                    {k: self._helpers.fingerprint_default(v) for k, v in kwd.items()},
                )
            except (TypeError, pickle.PicklingError, AttributeError, OverflowError) as e:
                return self._defaults_unhashable(func_name, pos, kwd, e)
        return self._finish_defaults_fold(func, state_hash, digest, pos, kwd, pinnable)

    def _defaults_unhashable(
        self,
        func_name: str,
        pos: tuple,
        kwd: dict,
        e: Exception,
    ) -> None:
        """Warn (once) that a default is unhashable; ``None`` = refuse to cache."""
        bad_type = self._args.first_unhashable_arg_type(pos, kwd)
        self._notices.warn_once(
            CashCacheIneffectiveWarning,
            func_name,
            bad_type,
            f"@cash.cache on {func_name}: a parameter default of type "
            f"{bad_type} could not be hashed ({type(e).__name__}), so the "
            f"function does not cache for any caller rather than risk "
            f"serving a result computed under a default that changed.",
            code="KEY-UNHASHABLE-DEFAULT",
            fix="get the value out of the signature -- build it in the body "
            "or require it at the call site"
            + (
                "."
                if isinstance(self._args.first_unhashable_arg(pos, kwd), CODE_VALUE_TYPES)
                else f" -- or register a hasher with cash.register_hasher({bad_type}, ...)."
            ),
        )
        return None

    def _finish_defaults_fold(
        self,
        func: Callable,
        state_hash: str,
        digest: str,
        pos: tuple,
        kwd: dict,
        pinnable: bool,
    ) -> str:
        """Memoize *digest* when it cannot drift, then mix it into *state_hash*."""
        # Two conditions gate the memo, and both are load-bearing:
        #
        # 1. Every default must be immutable. A mutable default is shared across
        #    calls and can be mutated in place (`def f(xs=[])`), so its content
        #    must be re-read every call; a memo would pin the first call's value
        #    and hand that result back forever.
        # 2. The callee must not wrap another function. The memo is validated
        #    against `func`'s OWN containers, which say nothing about defaults
        #    reached through `__wrapped__`; re-hashing those per call keeps the
        #    validation honest rather than merely cheap.
        if (
            pinnable
            and getattr(func, "__wrapped__", None) is None
            and all(is_immutable_capture(v) for v in pos)
            and all(is_immutable_capture(v) for v in kwd.values())
        ):
            try:
                self._defaults_pins[func] = (
                    getattr(func, "__defaults__", None),
                    dict(getattr(func, "__kwdefaults__", None) or {}),
                    digest,
                )
            except TypeError:
                pass  # not weak-referenceable; recompute per call
        return hashlib.sha256(f"{state_hash}:defaults:{digest}".encode("utf-8")).hexdigest()

    def fold_bound_self(
        self,
        func: Callable,
        func_name: str,
        state_hash: str,
    ) -> str:
        """Mix a bound method's instance state into the key.

        ``c.cache(obj.method)`` wraps an already-bound method: ``self`` never
        appears in ``args``, so two instances with different state shared one
        cache key and silently returned each other's results. Fold
        ``func.__self__`` through the same machinery as an ordinary argument
        (so ``register_hasher`` applies exactly as it does for in-class
        decoration, where ``self`` arrives via ``args``). Hashed per call,
        not at decoration: instance state may change between calls.

        Unhashable instances fall back to ``id(self)`` — correct (distinct
        instances stay distinct) but process-local; a one-shot warning points
        at ``register_hasher``.
        """
        if not inspect.ismethod(func):
            return state_hash
        owner = func.__self__
        try:
            self_hash = self._args.hash_payload((owner,), {})
        except (TypeError, pickle.PicklingError, AttributeError, OverflowError) as e:
            owner_type = type(owner).__name__
            self._notices.warn_once(
                CashCacheIneffectiveWarning,
                func_name,
                owner_type,
                f"@cash.cache on bound method {func_name}: the instance's state "
                f"could not be hashed ({type(e).__name__}), so cash fell back "
                f"to the instance's identity - entries are not shared across "
                f"equal instances and do not survive the process.",
                code="KEY-INSTANCE-STATE",
                fix=f"register a hasher with "
                f"cash.register_hasher({owner_type}, ...), returning "
                f"something derived from the state that affects the "
                f"result.",
            )
            self_hash = f"selfid:{id(owner)}"
        return hashlib.sha256(f"{state_hash}:boundself:{self_hash}".encode("utf-8")).hexdigest()

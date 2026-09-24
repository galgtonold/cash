"""Keying one intercepted call: :func:`call_cache_key` and the tests it rests on.

A call's key is the statement path's key (:func:`compute_cache_key`, under the
``"call"`` namespace) extended with the parts a call needs that a statement
does not: its computed arguments, the enclosing loops' variables, the globals
the callee writes, and -- when everything the call reads is plain data or
code (:func:`_keys_by_content`) -- the values it receives instead of the
statement it sits in.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import decimal as _decimal
import dis as _dis
import fractions as _fractions
import functools
import hashlib
import inspect as _inspect
import logging
import pathlib as _pathlib
import sys
import types as _types
from collections.abc import Callable, Mapping
from types import ModuleType as _ModuleType
from typing import TYPE_CHECKING, Any

from cash._memo import LruMemo
from cash.analysis.callee_effects import source_global_mutations
from cash.analysis.namespace_effects import capturable_globals
from cash.exceptions import SOURCE_RETRIEVAL_ERRORS
from cash.install_paths import is_user_path
from cash.notebook.cache_key import CacheKeyContext, compute_cache_key
from cash.object_hashing import compute_hash, compute_hash_full, pandas_nbytes

if TYPE_CHECKING:
    from cash.notebook.call_interception import CallSite

logger = logging.getLogger(__name__)

__all__ = ["CallKeys", "call_cache_key", "callee_mutated_globals", "global_digests", "global_names_reached"]


def _is_dunder_loop_var(name: str) -> bool:
    """True if *name*'s own bare portion is dunder-prefixed.

    Handles both shapes `loop_vars` can arrive in: a bare name (`"x"` --
    every direct caller/test that predates depth-keying, e.g.
    `test_call_unit_key.py`'s hand-built dicts) and a depth-prefixed one
    (`"0:x"` -- the production path,
    `CallRouting.current_loop_vars_for_call_key`). A depth-prefixed
    dunder (`"0:__iterable_lineage__"`) no longer starts with `"__"` itself
    once the prefix is on -- checking the combined string, as this used to,
    would silently stop catching it and turn the enforced reorder guard back
    into a merely documented one. Splitting off the depth at the first colon
    before checking is what keeps the guard live for both shapes.
    """
    _, _, bare = name.partition(":")
    return (bare or name).startswith("__")


def _loop_var_digest(name: str, value: object, loop_var_digests: Mapping[str, str]) -> str:
    """The discriminating hash for one loop-var entry: full, never sampled.

    A loop variable is the per-iteration discriminator, so it gets the full
    hash: the sampling ``compute_hash`` reduces a long list to its ends, and
    two items agreeing there would share one entry (a wrong value on the
    first run). The fallback must stay ``compute_hash_full``.

    A full hash of a large value can cost more than the call it keys, and is
    paid per call, so *loop_var_digests* -- the hashes ``for_handler`` took
    when it bound each loop variable, carried down ``CallRouting``'s
    ``loop_vars_scope`` stack -- are used when present. Not
    ``variable_lineage``: it is keyed by bare name and never popped, so after
    an inner loop reusing the outer loop's name it holds the inner value.
    """
    digest = loop_var_digests.get(name)
    return digest if digest is not None else compute_hash_full(value)


#: ``code object -> the globals its body mutates``, before the namespace filter.
#:
#: Keyed on the code object so that a hit skips ``inspect.getsource``, which is
#: the whole cost of the analysis and would otherwise be paid on every
#: intercepted call. A redefined function has a new code object, so an edit is
#: never served the old verdict. It is the code of the function ``getsource``
#: reads, after ``inspect.unwrap``: every function one ``functools.wraps``
#: decorator returns shares the wrapper's code.
_GLOBAL_MUTATION_CACHE: LruMemo[Any, tuple[str, ...]] = LruMemo(4096)


def callee_mutated_globals(fn) -> tuple[str, ...]:
    """Names in *fn*'s own globals that calling *fn* mutates in place, sorted.

    The same analysis the statement path applies to a callee it finds by name
    (:func:`~cash.analysis.callee_effects.source_global_mutations`), read from
    the live function object instead. Never raises: a callee whose source
    cannot be read (a C builtin, an ``exec``'d string, a partial) yields
    ``()``, so its writes are not captured -- as for any call cash cannot see
    into.
    """
    globals_dict = getattr(fn, "__globals__", None)
    if not isinstance(globals_dict, dict):
        return ()
    try:
        memo_key = getattr(_inspect.unwrap(fn), "__code__", None)
    except ValueError:  # a __wrapped__ cycle; getsource cannot read it either
        return ()
    cached = _GLOBAL_MUTATION_CACHE.get(memo_key) if memo_key is not None else None
    if cached is None:
        try:
            cached = tuple(sorted(source_global_mutations(_inspect.getsource(fn))))
        except SOURCE_RETRIEVAL_ERRORS:
            cached = ()
        if memo_key is not None:
            _GLOBAL_MUTATION_CACHE[memo_key] = cached
    # Filtered per call, not memoised: whether a name is bound (and not a
    # module) can change between calls, and the memo is about the source.
    capturable = capturable_globals(cached, globals_dict)
    return tuple(n for n in cached if n in capturable)


def call_cache_key(
    site: CallSite,
    *,
    ctx: CacheKeyContext,
    arg_digests: list[str],
    loop_vars: dict[str, object],
    loop_var_digests: Mapping[str, str] | None = None,
    global_digests: Mapping[str, str] | None = None,
    by_content: bool = False,
    name_digests: Mapping[str, str] | None = None,
) -> str | None:
    """The cache key for one intercepted call, or ``None`` to refuse caching it.

    Built by :func:`compute_cache_key` (the only place a key is assembled) from
    the *call's* source and free names, under the ``"call"`` namespace. The
    callee is one of the free names, so editing it re-keys the call and the
    globals it reaches are folded in; bare-name arguments resolve through the
    lineage ladder, a dict lookup per call. Then, each hashed and
    ``|``-delimited so no two parts can run together:

    * *arg_digests*: a content hash of each argument that is not a bare name,
      one per ``site.computed_arg_positions``. Required: ``compute(next(it))``
      reads an ``it`` whose lineage never moves, so without it every
      iteration would share one entry. A count mismatch returns ``None``;
      the caller then runs the call uncached.
    * *loop_vars*: the enclosing loops' variables, by value, keyed
      ``"{depth}:{name}"`` so a name reused by a nested loop keeps both.
      Values, not the iteration context: they are order-independent and
      stable across runs, so reordering an iterable or re-running only a
      loop's tail keys correctly. Dunder entries (``__iterable_lineage__``,
      which changes for every iteration on a reorder) are filtered out here,
      past the depth prefix. *loop_var_digests* are their precomputed full
      hashes (see :func:`_loop_var_digest`); leaving it out is correct, only
      slower.
    * *global_digests*: the pre-call state of each global the callee writes,
      so a call entered with a different accumulator is not served another's
      end state.
    * ``site.stmt_identity``: the enclosing statement, so two statements
      making the same call (``fetch_next(conn)`` in two cells' loops) keep
      their own entries.

    *by_content* keys the call on what it receives: the names read only in
    computed arguments, and the bare names in *name_digests* (full hashes of
    those arguments), leave the base, and the statement's identity leaves the
    key, so fixing some of a frame's rows re-runs only the calls whose inputs
    changed. Only the caller can say that is safe: :class:`CallUnit` does
    when everything the call reads is plain data or code.
    """
    free_names = set(site.free_names)
    source, occurrence = site.source, site.occurrence_index
    if by_content:
        free_names -= site.content_names
        free_names -= set(name_digests or ())
        # The call's shape, not its spelling: the computed arguments' values
        # are in the key. And no occurrence: the same call on the same values
        # twice in a cell is the same result, which is what content keying
        # already asserts across statements.
        if getattr(site, "content_source", ""):
            source, occurrence = site.content_source, 0
    base = compute_cache_key(
        source,
        free_names,
        ctx=ctx,
        occurrence_index=occurrence,
        namespace="call",
    ).cache_key
    # A collapsed key is a wrong answer; an uncached call is merely slow.
    if len(arg_digests) != len(site.computed_arg_positions):
        return None

    filtered_loop_vars = {name: value for name, value in loop_vars.items() if not _is_dunder_loop_var(name)}

    if not arg_digests and not filtered_loop_vars and not site.stmt_identity and not global_digests and not by_content:
        return base
    # Length-prefixed and `|`-delimited, so two calls cannot collide on a join.
    parts = [f"{len(arg_digests)}"]
    parts.extend(arg_digests)
    resolved_digests: Mapping[str, str] = loop_var_digests or {}
    parts.extend(
        f"{name}={_loop_var_digest(name, value, resolved_digests)}"
        for name, value in sorted(filtered_loop_vars.items())
    )
    # `g:`, `n:`: a global or a name cannot take a loop variable's slot.
    if global_digests:
        parts.extend(f"g:{name}={digest}" for name, digest in sorted(global_digests.items()))
    if by_content:
        parts.extend(f"n:{name}={digest}" for name, digest in sorted((name_digests or {}).items()))
        # Marked, so a key without the statement can never equal one with it.
        parts.append("by=content")
    elif site.stmt_identity:
        # Hashed, so the statement's own text cannot break the delimiting.
        parts.append("stmt=" + hashlib.sha256(site.stmt_identity.encode("utf-8")).hexdigest())
    return "call:" + hashlib.sha256((base + "|" + "|".join(parts)).encode("utf-8")).hexdigest()


#: Values whose content is all there is to them: nothing about them can change
#: while their hash stays put. The test :func:`_keys_by_content` applies to
#: everything a call reads before keying it without its statement.
_PLAIN_ATOMS = (
    int,
    float,
    complex,
    str,
    bytes,
    type(None),
    _decimal.Decimal,
    _fractions.Fraction,
    _dt.date,
    _dt.time,
    _dt.timedelta,
    _pathlib.PurePath,
    range,
)
#: How many values one call may have looked at before the answer is "not
#: plain" -- a long list is keyed the old way rather than walked.
_PLAIN_BUDGET = 10_000
#: Computed arguments are hashed in full under content keying; past this many
#: bytes the hash could cost more than the call saves, so the old key stays.
_CONTENT_KEY_MAX_BYTES = 64 * 1024 * 1024
#: An argument passed by name is hashed per call only up to this size; a
#: bigger one keeps its lineage, a dict lookup.
_NAME_CONTENT_MAX_BYTES = 1024 * 1024


def _is_plain_data(value, budget: list[int]) -> bool:
    """True when *value* is data whose content hash covers all of its state.

    Scalars, strings, dates, paths; numpy arrays and scalars without Python
    objects inside; pandas frames, series and indexes (hashed cell by cell);
    builtin containers of those. Anything else -- a connection, an iterator,
    a model, a user class -- may hold state its key cannot see.
    """
    budget[0] -= 1
    if budget[0] < 0:
        return False
    if isinstance(value, _PLAIN_ATOMS):
        return True
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(value, np.ndarray):
            return not value.dtype.hasobject
        if isinstance(value, np.generic):
            return not isinstance(value, np.object_)
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
        return True
    if type(value) in (list, tuple, set, frozenset):
        return all(_is_plain_data(item, budget) for item in value)
    if type(value) is dict:
        return all(_is_plain_data(k, budget) and _is_plain_data(v, budget) for k, v in value.items())
    return False


@functools.lru_cache(maxsize=1024)
def _code_names(code: _types.CodeType) -> frozenset[str]:
    """Global names *code* and the functions nested in it load.

    From the bytecode, not ``co_names``, which holds attribute names too:
    ``os.open`` would read as the global ``open`` -- cash's file-tracking
    wrapper in a notebook.
    """
    names = {ins.argval for ins in _dis.get_instructions(code) if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME")}
    for const in code.co_consts:
        if isinstance(const, _types.CodeType):
            names |= _code_names(const)
    return frozenset(names)


def _plain_or_code(value, seen: set[int], budget: list[int]) -> bool:
    if isinstance(value, _types.FunctionType):
        if getattr(value, "_is_file_tracker_patch", False) or getattr(value, "_cash_cached", False):
            return True  # cash's own: keyed or tracked by cash itself

        filename = getattr(value.__code__, "co_filename", "") or ""
        # A cell's code has a `<cash-...>` / `<ipython-...>` name: the user's.
        if filename and not filename.startswith("<") and not is_user_path(filename):
            # A library's function is code, as its classes are. Its module
            # state is no more in a lineage key than in a content key, and
            # walking it refused: sklearn's `normalize` is a validating
            # wrapper whose globals hold non-plain state, so `fit_vectors`
            # re-fitted on byte-identical text.
            return True
        return _callee_state_is_plain(value, seen, budget)
    if isinstance(value, (_ModuleType, type, _types.BuiltinFunctionType)):
        return True
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.ufunc):
        return True
    return _is_plain_data(value, budget)


def _callee_state_is_plain(fn, seen: set[int], budget: list[int]) -> bool:
    """True when every global, closure cell and default *fn* can reach --
    through the functions it calls too -- is plain data or code."""
    if id(fn) in seen:
        return True
    seen.add(id(fn))
    namespace = getattr(fn, "__globals__", None) or {}
    for name in _code_names(fn.__code__):
        if name in namespace and not _plain_or_code(namespace[name], seen, budget):
            return False
    for cell in fn.__closure__ or ():
        try:
            contents = cell.cell_contents
        except ValueError:  # an empty cell: nothing bound yet
            continue
        if not _plain_or_code(contents, seen, budget):
            return False
    defaults = (*(fn.__defaults__ or ()), *(fn.__kwdefaults__ or {}).values())
    return all(_is_plain_data(value, budget) for value in defaults)


def _nbytes(value) -> int:
    """Roughly how many bytes a full content hash of *value* reads."""
    np = sys.modules.get("numpy")
    if np is not None and isinstance(value, np.ndarray):
        return int(value.nbytes)
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(value, (pd.DataFrame, pd.Series)):
        sized = pandas_nbytes(value)
        if sized is not None:
            return int(sized)
        usage = value.memory_usage(index=True, deep=False)
        return int(usage.sum() if hasattr(usage, "sum") else usage)
    if isinstance(value, (str, bytes)):
        return len(value)
    return 0


def global_names_reached(fn, seen: set[int] | None = None, depth: int = 0) -> set[str]:
    """Global names *fn* loads, and those of the functions it reaches, bounded."""
    seen = set() if seen is None else seen
    code = getattr(fn, "__code__", None)
    if code is None or id(fn) in seen or depth > 6:
        return set()
    seen.add(id(fn))
    names = set(_code_names(code))
    namespace = getattr(fn, "__globals__", None) or {}
    for name in list(names):
        value = namespace.get(name)
        if isinstance(value, _types.FunctionType):
            names |= global_names_reached(value, seen, depth + 1)
        elif isinstance(value, type) and id(value) not in seen:
            # A class the callee builds or calls into: its methods read globals too.
            seen.add(id(value))
            for member in vars(value).values():
                member = getattr(member, "__func__", member)
                if isinstance(member, _types.FunctionType):
                    names |= global_names_reached(member, seen, depth + 1)
    return names


def _loop_vars_the_call_can_read(
    fn, site: CallSite, loop_vars: Mapping[str, object], name_digests: Mapping[str, str] | None
) -> dict[str, object]:
    """The enclosing loops' variables a content-keyed call can still read
    without them being in its key.

    A loop variable is in a call's key to close a channel the arguments do not
    cover: hidden state behind a name, or a global the callee reads. Keyed on
    what it receives, a call's arguments are hashed by value, so a loop
    variable passed in (`make_features(cleaned[mid], win)`) is already there,
    and one the call never reads cannot change its result. Kept anyway, it
    made the sweep's keys differ from the same call outside a loop: scoring
    with the chosen window re-fitted all 200 machines the sweep had just fitted.
    Kept: a loop variable the callee, or a function it
    calls, reads as a global; and one named in the call itself but not hashed
    by value (an argument too big to hash per call). Only a variable the
    arguments carry is dropped: one the call does not mention stays, since
    what the callee can reach is followed through functions and classes but
    not every path (a dispatch table, an object's attribute).
    """
    hashed = set(name_digests or ()) | set(site.content_names)
    reached = None
    kept = {}
    for key, value in loop_vars.items():
        bare = key.split(":", 1)[1] if ":" in key else key
        if bare in hashed:
            if reached is None:
                reached = global_names_reached(fn)
            if bare not in reached:
                continue
        kept[key] = value
    return kept


def _keys_by_content(fn, site: CallSite, args: tuple, kwargs: dict, loop_vars: Mapping[str, object]) -> bool:
    """Whether the call may be keyed on what it receives (see
    :func:`call_cache_key`'s *by_content*).

    Only when nothing it reads can change while its key stays put: every
    argument, every loop variable around it, and everything the callee
    reaches is plain data or code, and the arguments to hash in full are
    small enough to be worth it.
    """
    combined = (*args, *kwargs.values())
    budget = [_PLAIN_BUDGET]
    if not all(_is_plain_data(value, budget) for value in combined):
        return False
    if not all(_is_plain_data(value, budget) for name, value in loop_vars.items() if not _is_dunder_loop_var(name)):
        return False
    computed = sum(_nbytes(combined[pos]) for pos in site.computed_arg_positions if pos < len(combined))
    if computed > _CONTENT_KEY_MAX_BYTES:
        return False
    return _callee_state_is_plain(fn, set(), budget)


def global_digests(fn, names: tuple[str, ...]) -> dict[str, str]:
    """PRE-call content hashes of the globals *fn* writes, for the key.

    ``compute_hash_full``, never the sampling ``compute_hash``, for the
    same reason :func:`_loop_var_digest` documents at length: this IS the
    discriminator. ``compute_hash`` reduces a collection over 200 elements
    to its first and last five, so two different accumulator states that
    agree at both ends would key IDENTICALLY -- and an accumulator is
    precisely the shape that grows in the middle. That is first-run
    wrongness, not a missed optimisation.

    The cost this admits is real and bounded by how rare the case is: a
    callee that writes a global at all is uncommon, and the hash is over
    the accumulator, not over the arguments. ``hash_args``' sampling trade
    is fine where it lives (a coarse per-call mutation smoke test on a
    possibly-huge live argument, allowed to be wrong toward "assume
    unmutated"); it is not fine here.

    A name that cannot be hashed at all is omitted, which makes the key
    LESS discriminating -- so ``call_effects.capture_globals`` independently
    refuses to store any entry whose capture is not sound, and the pair of
    them fails closed.
    """
    globals_dict = getattr(fn, "__globals__", None) or {}
    digests: dict[str, str] = {}
    for name in names:
        try:
            digests[name] = compute_hash_full(globals_dict[name])
        except Exception:  # noqa: BLE001 - a missing digest only widens the key
            logger.debug("call unit: could not digest global %r", name)
    return digests


class CallKeys:
    """Builds each intercepted call's key from the live processor state, and
    remembers what each call site was keyed on so a miss can say what moved.

    The providers are read at key time, not captured: one ``CallCache``
    serves every statement, and each brings its own lineage and loop.
    """

    def __init__(
        self,
        ctx_provider: Callable[[], CacheKeyContext],
        loop_vars_provider: Callable[[], dict[str, object]] | None = None,
        loop_var_digests_provider: Callable[[], Mapping[str, str]] | None = None,
    ):
        self._ctx_provider = ctx_provider
        # See `call_cache_key`'s `loop_vars` section. `None` means "no loop
        # context available", which degrades to `{}`: the key a call gets
        # outside any loop.
        self._loop_vars_provider = loop_vars_provider or (lambda: {})
        # See `call_cache_key`'s `loop_var_digests` section and
        # `_loop_var_digest`'s docstring. `None` is always CORRECT (every
        # entry falls through to a fresh `compute_hash_full`), only slower.
        self._loop_var_digests_provider = loop_var_digests_provider or (lambda: {})
        # Per call SITE: what its key was built from last time, and what moved
        # since -- the badge's answer to "why did this re-run?".
        self._site_parts: dict[tuple, tuple[str, dict]] = {}
        self._site_reason: dict[tuple, str | None] = {}
        # Sites already keyed in THIS statement run. A site called once per
        # loop item keys differently per item BY DESIGN -- that is the loop
        # variable doing its job, not something to explain. Only the first
        # call of each run is compared, against the first call of the last.
        self._keyed_this_run: set = set()

    def begin_statement(self) -> None:
        """A new statement run: each site's first call is compared again."""
        self._keyed_this_run.clear()

    def key(
        self, site: CallSite, args: tuple, kwargs: dict, global_digests: Mapping[str, str] | None = None, fn=None
    ) -> str | None:
        """The call's key. With *fn*, keyed on what it receives when
        :func:`_keys_by_content` allows it."""
        if site.has_unpacking and fn is not None:
            return self._build_unpacked_key(site, args, kwargs, fn)
        if site.has_unpacking:
            # `*args`/`**kwargs` unpacking means the call's live arity is not
            # statically known. `site.computed_arg_positions` is a STATIC
            # count (every position, fail-closed -- see
            # `_computed_arg_positions`), which need not match the RUNTIME
            # flattened `(*args, *kwargs.values())` length: `compute(*pair())`
            # has one static position but the pair unpacks to two live
            # arguments, and indexing only position 0 would hash the first
            # element and silently ignore the rest (reproduced as a second, DIFFERENT pair() result being served the
            # first call's cached value). Refuse the whole site rather than
            # mint a key that looks discriminated but isn't; an uncached call
            # is merely slow.
            return None
        try:
            loop_vars = self._current_loop_vars()
            by_content = fn is not None and _keys_by_content(fn, site, args, kwargs, loop_vars)
            if getattr(site, "in_loop_unit", False) and not by_content:
                return None
            arg_digests = self._arg_digests(site, args, kwargs, full=by_content)
            name_digests = self._name_digests(site, args, kwargs) if by_content else None
            if by_content and loop_vars:
                loop_vars = _loop_vars_the_call_can_read(fn, site, loop_vars, name_digests)
            ctx = self._ctx_provider()
            key = call_cache_key(
                site,
                ctx=ctx,
                arg_digests=arg_digests,
                loop_vars=loop_vars,
                loop_var_digests=self._current_loop_var_digests(),
                global_digests=global_digests,
                by_content=by_content,
                name_digests=name_digests,
            )
            self._note_key_parts(site, key, ctx, arg_digests, global_digests)
            return key
        except Exception:  # noqa: BLE001 - never let keying break the call
            logger.debug("call unit: key build failed for %s", site.source)
            return None

    def _build_unpacked_key(self, site: CallSite, args: tuple, kwargs: dict, fn) -> str | None:
        """The key of a call with ``*``/``**`` unpacking, keyed on what it received.

        ``fit_series(g, **TUNED.get(dept, {}))`` in a comprehension ran uncached:
        positions written in the source say nothing about the values
        that arrive, so the site was refused (``compute(*pair())``
        had keyed the first of two values). What did arrive is in hand here:
        every positional value, and every keyword with its name, hashed in
        full. Only when the call may be keyed on content at all
        (:func:`_keys_by_content`); otherwise refused as before.
        """
        try:
            count = len(args) + len(kwargs)
            received = dataclasses.replace(
                site,
                computed_arg_positions=tuple(range(count)),
                local_arg_positions=tuple(range(count)),
                name_arg_positions=(),
                has_unpacking=False,
            )
            loop_vars = self._current_loop_vars()
            if not _keys_by_content(fn, received, args, kwargs, loop_vars):
                return None
            digests = [compute_hash_full(value) for value in args]
            digests.extend(f"{name}:{compute_hash_full(kwargs[name])}" for name in sorted(kwargs))
            if loop_vars:
                loop_vars = _loop_vars_the_call_can_read(fn, received, loop_vars, None)
            return call_cache_key(
                received,
                ctx=self._ctx_provider(),
                arg_digests=digests,
                loop_vars=loop_vars,
                loop_var_digests=self._current_loop_var_digests(),
                by_content=True,
            )
        except Exception:  # noqa: BLE001 - never let keying break the call
            logger.debug("call unit: key build failed for %s", site.source)
            return None

    def _current_loop_vars(self) -> dict[str, object]:
        """The live enclosing loop's non-dunder iteration vars, or ``{}``.

        Wired to ``CallRouting.current_loop_vars_for_call_key`` (see that class's
        ``_loop_vars`` stack, pushed/popped by
        ``ForLoopHandler._process_one_iteration`` around each iteration's body)
        via ``CallCache``'s ``loop_vars_provider``. Guarded independently of
        ``key``'s own try/except: a provider failure should degrade to
        "no loop discriminator" ``{}`` -- same as running outside a loop --
        not to refusing the key (and therefore the call's caching) entirely.
        """
        try:
            loop_vars = self._loop_vars_provider()
        except Exception:  # noqa: BLE001 - degrade, don't refuse the whole key
            logger.debug("call unit: loop_vars_provider failed for this call")
            return {}
        return loop_vars if isinstance(loop_vars, dict) else {}

    def _current_loop_var_digests(self) -> Mapping[str, str]:
        """The live enclosing loop's precomputed loop-var digests, or ``{}``.

        Wired to ``CallRouting.current_loop_var_digests_for_call_key`` (see that
        class's ``_loop_var_digests`` stack -- pushed/popped in
        lockstep with ``_loop_vars``, by the same
        ``loop_vars_scope`` call) via ``CallCache``'s
        ``loop_var_digests_provider``. Guarded independently of
        ``key``'s own try/except, same reasoning as
        ``_current_loop_vars``: a provider failure degrades to "no
        precomputed digest available" ``{}``, which ``_loop_var_digest``
        treats as "fall through to a fresh `compute_hash_full`" -- slower,
        never wrong -- not to refusing the key entirely.
        """
        try:
            digests = self._loop_var_digests_provider()
        except Exception:  # noqa: BLE001 - degrade, don't refuse the whole key
            logger.debug("call unit: loop_var_digests_provider failed for this call")
            return {}
        return digests if isinstance(digests, Mapping) else {}

    @staticmethod
    def _name_digests(site: CallSite, args: tuple, kwargs: dict) -> dict[str, str]:
        """Full hashes of the small arguments passed as a bare name (see
        :func:`call_cache_key`'s *name_digests*). Past
        ``_NAME_CONTENT_MAX_BYTES`` a value keeps its lineage instead."""
        combined = (*args, *kwargs.values())
        digests = {}
        for name, pos in getattr(site, "name_arg_positions", ()):
            if pos < len(combined) and _nbytes(combined[pos]) <= _NAME_CONTENT_MAX_BYTES:
                digests[name] = compute_hash_full(combined[pos])
        return digests

    def _arg_digests(self, site: CallSite, args: tuple, kwargs: dict, full: bool = False) -> list[str]:
        """Content hashes of the live arguments at ``site.computed_arg_positions``.

        Positions are in ``(*args, *kwargs.values())`` order, matching how
        :func:`_computed_arg_positions` numbered them at rewrite time. A
        position beyond the live call's arity (the wrapped function called with
        a different shape than the site predicted) is simply not appended --
        the resulting length mismatch is caught by ``call_cache_key`` itself,
        which refuses rather than mint a key with a discriminator missing.

        *full* hashes every one of them in full: under content keying the
        value is all the key knows of where the argument came from.
        """
        combined = (*args, *kwargs.values())
        local = set(getattr(site, "local_arg_positions", ()))
        digests = []
        for pos in site.computed_arg_positions:
            if pos >= len(combined):
                continue
            # A comprehension's own variable discriminates its elements, as a
            # loop variable does its iterations: full hash, never sampled.
            hash_fn = compute_hash_full if full or pos in local else compute_hash
            digests.append(hash_fn(combined[pos]))
        return digests

    def _note_key_parts(self, site: CallSite, key, ctx, arg_digests, global_digests) -> None:
        """Remember what this call site was keyed on, and what moved since the
        last time it was keyed (read back by :meth:`why_missed`).

        In-memory and per site, like the statement guard's own components. A
        first run in a fresh kernel has nothing to compare against and says
        nothing -- the same deliberate silence a statement keeps, since "first
        time" is self-evident to someone running a cell for the first time.
        """
        if key is None:
            return
        try:
            site_id = self._site_id(site)
            if site_id in self._keyed_this_run:
                return
            self._keyed_this_run.add(site_id)
            lineages = getattr(ctx, "variable_lineage", None) or {}
            parts = {name: lineages.get(name) for name in site.free_names}
            for i, digest in enumerate(arg_digests or ()):
                parts[f"argument {i + 1}"] = digest
            for name, digest in (global_digests or {}).items():
                parts[name] = digest
            seen = self._site_parts.get(site_id)
            self._site_parts[site_id] = (key, parts)
            if not seen or seen[0] == key:
                self._site_reason.pop(site_id, None)
                return
            moved = sorted(name for name in set(parts) | set(seen[1]) if parts.get(name) != seen[1].get(name))
            # The key moved with no named part of it moving: something else did
            # (a file the callee reads, the callee's own source). Naming
            # nothing beats naming the wrong thing.
            named = ", ".join(moved[:3]) + (", ..." if len(moved) > 3 else "")
            self._site_reason[site_id] = f"changed: {named}" if moved else None
        except Exception:  # noqa: BLE001 - attribution never breaks a call
            logger.debug("call unit: could not note key parts for %s", site.source)

    @staticmethod
    def _site_id(site: CallSite) -> tuple[str, int, str]:
        return (site.source, site.occurrence_index, getattr(site, "stmt_identity", ""))

    def why_missed(self, site: CallSite) -> str | None:
        """Which named part of this call's key moved since it was last keyed.

        A sweep re-ran and the badge said only "0/6 hit", so the user had to
        guess why -- and guessed wrong, then reported the re-run as a
        suspected bug.
        """
        try:
            return self._site_reason.get(self._site_id(site))
        except Exception:  # noqa: BLE001
            return None

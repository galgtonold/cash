"""The one formula for a variable's lineage, shared by both engines.

The runtime (``statement/lineage.py``) records a variable's lineage when its
statement runs; the upstream simulator (``upstream/virtual_lineage.py``)
recomputes it from the code above a cell. Wherever the two disagree, a cell
reading the variable sees "changed" with nothing changed and re-runs other
cells' statements to repair it, so the two build it from the functions here.

Everything here is a pure function of its arguments. The callers decide which
ingredients they have; this module decides how they combine.
"""

from __future__ import annotations

import ast
import builtins
import functools
import hashlib
import logging
import os
import pickle
import secrets
import sys
import types
from collections.abc import Mapping
from typing import Any, Callable, Iterable

from cash._user_attrs import attribute_or, has_attribute, instance_of, is_routine

from .._memo import CODE_OBJECTS, NOTEBOOK_STATEMENTS, LruMemo
from ..analysis.ast_util import parse_cached, walk_tree
from ..analysis.mutations import repeatable_setting_receiver
from ..analysis.purity_analyzer import get_analyzer
from ..code_digest import module_identity
from ..effects import ENVIRONMENT_TEXT_MARKERS, environment_component, environment_input
from ..source_norm import exact_source_digest, unparse_without_docstrings
from ..tracking.module_symbols import closure_digest, static_attribute_reads
from ..tracking.randomness import drawn_rng_vars, hidden_lineage_reads, observed_rng_reads
from ..value_hash import budgeted_hash, compute_hash, is_identity_fallback_hash
from .callee_reach import Reach, reached_user_code
from .held_code import held_code_digest
from .read_handles import read_handle_digest

logger = logging.getLogger(__name__)


def is_cash_instrumentation(val: object) -> bool:
    """True when *val* is I/O instrumentation rather than a notebook value:
    one of cash's own I/O-tracking wrappers, or the ``open`` IPython puts in
    ``user_ns``.

    Neither can be pickled, so ``compute_hash`` falls back to a memory
    address and every statement mentioning one would get a per-kernel key and
    lineage. In a plain interpreter ``open`` is a builtin and a wrapped reader
    stands for the library's own function, so neither is an input.

    Tested with ``is True``, not truthiness: an object with a permissive
    ``__getattr__`` (``MagicMock``, RPC proxies) has a truthy attribute for
    any name, and treating it as instrumentation would drop a real input.
    """
    if attribute_or(val, "_is_file_tracker_patch", False) is True:
        return True
    shell = sys.modules.get("IPython.core.interactiveshell")
    return shell is not None and val is attribute_or(shell, "_modified_open", _NO_SHELL_OPEN)


_NO_SHELL_OPEN = object()


def is_module_like(var_name: str, val: object, virtual_modules: Iterable[str]) -> bool:
    """Whether *val* is valued as a module: by its tracked lineage alone, never
    by its content.

    A module (or a bound method, or an IPython internal) cannot be pickled, so
    ``compute_hash`` falls back to its memory address, which changes with
    every kernel. The cache key, the runtime's output lineage and the
    simulation's must all apply this same test, or a restart re-keys
    everything downstream.
    """
    if var_name in virtual_modules:
        return True
    if val is None:
        return False
    try:
        if instance_of(val, types.ModuleType):
            return True
        # IPython internals and bound methods behave like modules -- skip them
        if callable(val) and (var_name.startswith("_") or has_attribute(val, "__self__")):
            return True
    except (AttributeError, TypeError) as exc:
        logger.debug("[CACHE_KEY] Failed to check module/callable type for '%s': %s", var_name, exc)
    return False


def bound_names(code: str) -> set[str] | None:
    """The names *code* binds: assignment targets and import aliases; None
    when it does not parse."""
    tree = parse_cached(code)
    if tree is None:
        return None
    bound: set[str] = set()
    for node in walk_tree(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
    return bound


def changed_module_environment(
    name: str, value: Any, code: str, environment: str, virtual_modules: Iterable[str] = ()
) -> str:
    """*environment* (``recorded_reads_lineage_component``) for output
    *name*, or nothing when *name* is a module the statement changes in
    place (``mylib.K = 7``, ``mylib.set_k(7)``) rather than binds.

    Such a statement reads the data it changes (``mylib.CONFIG`` before
    ``mylib.CONFIG["k"] = 7``), which the run before left as it pleased: the
    module's lineage would then differ between two runs of the same
    notebook, and every reader of the module with it. Its lineage records
    which statements changed it and from what inputs, for a rebuild after a
    reload; what the module's data holds, its readers key on themselves.
    Both engines ask here.
    """
    if not (instance_of(value, types.ModuleType) or (value is None and name in virtual_modules)):
        return environment
    bound = bound_names(code)
    return environment if bound is None or name in bound else ""


def read_module_source_hash(mod_file: str, dep_files: set[str] | None = None) -> str | None:
    """Combined identity hash of a module file and its dependency files.

    See `cash.code_digest.module_identity` for what "identity" covers and why
    it is not the file's bytes.
    """
    own = module_identity(mod_file)
    if own is None:
        logger.debug("[MODULE_HASH] Could not read module file: %s", mod_file)
        return None
    if not dep_files:
        return own
    hasher = hashlib.sha256()
    hasher.update(own.encode("utf-8"))
    for dep_path in sorted(dep_files):
        dep = module_identity(dep_path)
        if dep is None:
            logger.debug("[MODULE_HASH] Could not read dependency file: %s", dep_path)
            continue
        hasher.update(dep.encode("utf-8"))
    return hasher.hexdigest()


def module_read_lineage(
    function_tracker: Any,
    var_name: str,
    value: Any,
    code: str | None,
) -> str | None:
    """What a statement reading module *var_name* depends on, narrowed to the
    names it reads -- or None, meaning the module's whole lineage, as before.

    A statement reading ``lib.load`` was keyed on the whole of ``lib``, so
    editing any other function in the file re-ran it and everything built on
    it: editing one helper re-read all 10,000
    ticket files, 48.7 s against a 17.3 s control, later 9.1x. This returns
    a digest of exactly what ``load`` reaches inside the module (see
    ``module_symbols``), plus the module's tracked dependency FILES whole --
    what `load` does can depend on another local module, and those files are
    what the whole-module digest folded in for that; per-symbol keying within
    one module changes nothing across modules.

    ONE function for all three places a module input is valued, because
    they must agree byte for byte: the cache key (``cache_key``), the output
    lineage the runtime records (``statement/lineage.py``), and the one the
    simulation recomputes (``upstream/virtual_lineage.py``). If the key alone
    narrowed, ``DATA = lib.load(6)`` would hit and still get a new lineage,
    and everything downstream of it would miss anyway.

    None -- keep the whole lineage -- whenever narrowing is not safe or not
    possible: the module is not live, not a tracked local file, the statement
    uses the module other than by plain attribute reads, or the closure
    cannot be bounded. None is always correct; it is only slower.
    """
    if function_tracker is None or not code or not instance_of(value, types.ModuleType):
        return None
    mod_file = attribute_or(value, "__file__", None)
    real = attribute_or(value, "__name__", var_name)
    names = {real, var_name}
    if not (mod_file and os.path.isfile(mod_file) and names & _tracked(function_tracker)):
        return None

    attrs = static_attribute_reads(code, var_name)
    if not attrs:
        return None
    return _closure_with_deps(function_tracker, mod_file, names, attrs, "symread:" + real)


def output_lineage(
    source_hash: str,
    input_lineages: Iterable[str],
    file_component: str = "",
    func_component: str = "",
    module_component: str = "",
    environment: str = "",
    value_component: str = "",
) -> str:
    """The lineage hash of one output of a statement.

    A lineage whose value was seen to equal an earlier one's
    (`value_cutoff`) is that one's: the runtime and the simulation both
    reach it through here."""
    lineage_str = (
        f"{source_hash}:{':'.join(sorted(input_lineages))}{file_component}{func_component}{module_component}"
        f"{environment}{value_component}"
    )
    return resolve_lineage_alias(hashlib.sha256(lineage_str.encode("utf-8")).hexdigest())


def binding_source_hash(source_hash: str, code: str, name: str) -> str:
    """The code half of *name*'s lineage when *code* bound it: *source_hash*,
    the statement's, except for an import, where it is the one clause that
    bound *name*.

    ``import pandas as pd, json`` and ``import json, pandas as pd`` bind the
    same ``pd``, but hashed whole they gave it two lineages, and an agent
    that wrote its imports in another order re-ran every ``pd.read_csv``
    after it. The clause keeps all that decides the binding (the module, its
    dots, the alias, a ``from`` and its level), and the module's own source
    is still in the module component. A name no clause binds by name (``from
    m import *``) keeps the statement's hash. ``import pandas as pd`` alone
    is its own clause, so its lineage is what it was.
    """
    tree = parse_cached(code)
    if tree is None or not tree.body or not all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body):
        return source_hash
    clause: ast.stmt | None = None
    for node in tree.body:
        for alias in node.names:
            bound = alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
            if bound != name:
                continue
            if isinstance(node, ast.Import):
                clause = ast.Import(names=[alias])
            else:
                clause = ast.ImportFrom(module=node.module, names=[alias], level=node.level)
    if clause is None:
        return source_hash
    narrowed = ast.unparse(ast.fix_missing_locations(ast.Module(body=[clause], type_ignores=[])))
    return hashlib.sha256(unparse_without_docstrings(narrowed).encode("utf-8")).hexdigest()


def setting_output_lineage(
    name: str,
    value: Any,
    code: str,
    input_lineages: Mapping[str, str],
    lineage_of: Callable[[list[str]], str],
    virtual_modules: Iterable[str] = (),
) -> str | None:
    """The lineage of module *name* after the statement *code* set it with a
    setting that is the same when run again (`repeatable_setting_receiver`),
    or None for the ordinary formula.

    The ordinary formula folds the module's lineage before the statement, so
    each run of ``pd.set_option('display.width', 120)`` moved ``pd`` on, and
    a session that ran the setup cell three times keyed every reader of
    ``pd`` apart from a Run All after a restart, which ran it once. Here the
    setting stands for itself: *lineage_of* builds the ordinary formula from
    the other inputs' lineages (*input_lineages* less the module's own), and
    :func:`setting_lineage` adds it to the settings the module had before.
    ONE function for the runtime (``statement/lineage.py``) and the
    simulation (``upstream/statement_lineage.py``).
    """
    if not (instance_of(value, types.ModuleType) or (value is None and name in virtual_modules)):
        return None
    if repeatable_setting_receiver(parse_cached(code)) != name:
        return None
    others = [lineage for input_name, lineage in input_lineages.items() if input_name != name]
    return setting_lineage(input_lineages.get(name), lineage_of(others))


#: A lineage :func:`setting_lineage` made -> ``(the lineage the settings
#: were made on, the settings in order)``, for this process. A lineage not
#: here starts a new run of settings; that is never wrong, only a miss.
_SETTING_RUNS: dict[str, tuple[str, tuple[str, ...]]] = {}


def setting_lineage(before: str | None, setting: str) -> str:
    """The lineage of a module after *setting* (a setting call's own
    lineage) on a module whose lineage was *before*.

    The module's lineage is the lineage it had before its run of settings
    and the settings in order. A run that ends in the same settings twice
    (``A B A B``) keeps them once (``A B``): each setting writes its values
    over what was there, so the second time changes nothing, and re-running
    the setup cell leaves the module's lineage where one run left it. Any
    other order is a different lineage (``A B`` then ``A``), even where the
    two settings touch different options: which ones a call touches
    (``plt.style.use`` sets dozens) is not known, and a miss is safe.
    """
    base, run = _SETTING_RUNS.get(before or "", (before or "", ()))
    run = _without_repeat((*run, setting))
    lineage = hashlib.sha256(f"settings:{base}:{':'.join(run)}".encode("utf-8")).hexdigest()
    _SETTING_RUNS[lineage] = (base, run)
    return lineage


def _without_repeat(run: tuple[str, ...]) -> tuple[str, ...]:
    """*run* with each block that ends it twice in a row kept once."""
    collapsed = True
    while collapsed:
        collapsed = False
        for size in range(1, len(run) // 2 + 1):
            if run[-size:] == run[-2 * size : -size]:
                run = run[:-size]
                collapsed = True
                break
    return run


#: Lineage -> the lineage of an earlier value with the same content
#: (`value_cutoff.ValueCutoff`), for this process.
_LINEAGE_ALIASES: dict[str, str] = {}


def resolve_lineage_alias(lineage: str) -> str:
    """*lineage*, or the earlier lineage its value was found equal to."""
    return _LINEAGE_ALIASES.get(lineage, lineage)


def record_lineage_alias(lineage: str, earlier: str) -> None:
    """Name the value of *lineage* by *earlier* from now on, in this process
    and, through the store `use_lineage_alias_store` set, the next."""
    earlier = resolve_lineage_alias(earlier)
    if lineage != earlier:
        _LINEAGE_ALIASES[lineage] = earlier
        if _ALIAS_STORE is not None:
            _ALIAS_STORE.record(lineage, earlier)


_ALIAS_STORE: Any = None


def use_lineage_alias_store(store: Any) -> None:
    """Read the aliases *store* kept (``value_cutoff.LineageAliasStore``),
    and keep the new ones there: an entry stored under an alias is found by
    the lineage a restarted kernel computes only if it knows the alias too."""
    global _ALIAS_STORE
    _ALIAS_STORE = store
    if store is not None:
        for lineage, earlier in store.aliases().items():
            _LINEAGE_ALIASES.setdefault(lineage, earlier)


def held_lineage(name: str, before: str, cache_key: str, resolve: bool = True) -> str:
    """The lineage of *name*, a variable stored with a statement's outputs
    because it holds one of their objects too
    (``StatementCacheMetadata.holders``), from its lineage *before* the
    statement and the statement's *cache_key*.

    The statement may change what the variable reaches (``b = a`` before
    ``a['x'] = ...``), so it moves on, as a derivation alias of a mutated base
    does, and the next statement sharing the object cannot be served the
    entry written before this change.

    Unless the statement was seen to leave the variable's content as it was
    (`keep_held_lineage`): then it keeps *before*. Not with *resolve* false:
    for a run that saw the content change, which an earlier run of the same
    statement, keyed alike, did not.
    """
    held = _held_hash(before, cache_key)
    if not resolve:
        return held
    kept_as = _kept_held(name, held)
    earlier = resolve_lineage_alias(kept_as)
    return earlier if earlier != kept_as else held


def keep_held_lineage(name: str, before: str, cache_key: str) -> None:
    """Note that the statement with *cache_key* left *name*, a variable it
    holds with lineage *before*, as it was, so `held_lineage` keeps *before*
    for it, in the runtime, the simulation and, through the alias store, the
    next session.
    ``pre.fit_transform(X)`` fits clones of the pipelines a ColumnTransformer
    holds; each fit moved those pipelines on all the same, so a session that
    fitted twice keyed everything built on them apart from a Run All after a
    restart, which fitted once, and 400 s of model fits ran again.

    The alias is *name*'s own (`_kept_held`): two holders can have one
    lineage (``c`` and ``o``, bound by one loop). Keyed by the lineage, the
    statement that left ``c`` as it was and changed ``o`` kept ``o`` too in
    the simulation and the restore after a restart, where the run had moved
    it on, and a reader of ``o`` was served what it gave before the change."""
    record_lineage_alias(_kept_held(name, _held_hash(before, cache_key)), before)


def _held_hash(before: str, cache_key: str) -> str:
    return hashlib.sha256(f"held:{before}:{cache_key}".encode("utf-8")).hexdigest()


def _kept_held(name: str, held: str) -> str:
    """What `keep_held_lineage` keys the alias of *name*, kept where it would
    have moved on to *held*, by: the moved lineage itself stays the same for
    every name, the alias is this one's alone."""
    return hashlib.sha256(f"kept-held:{name}:{held}".encode("utf-8")).hexdigest()


def no_cache_value_digest(value: Any) -> str:
    """The digest of what a ``# @cash:no-cache`` statement bound, for its
    output's lineage.

    Such a statement runs every run and can bind a new value with the same
    source and inputs, so its output's lineage carries the value: a reader
    below misses when the value changed and still hits when it did not. A
    value whose content cannot be hashed, or not within the budget of a
    change check (`budgeted_hash`), gets a digest of its own each run, so
    its readers always recompute: the identity fallback stays the same under
    in-place change, and a new object can reuse an old one's address. A file
    open for reading is named by its file and position instead
    (:func:`read_handle_digest`), so what is read from it can be cached.
    """
    handle = read_handle_digest(value)
    if handle is not None:
        return handle
    # Within the budget of a check whether a value changed: a list of 8
    # million tuples took 8 s to hash on every run. Past it, the value is
    # taken as new, as one that cannot be hashed is.
    digest = _function_digest(value) if type(value) is types.FunctionType else budgeted_hash(value)
    if digest is None or is_identity_fallback_hash(value, digest):
        return "fresh:" + secrets.token_hex(16)
    return digest


def _function_digest(fn: types.FunctionType, depth: int = 0) -> str | None:
    """A function by what it runs and holds: its code (`held_code_digest`)
    and the values of its defaults and closure cells, a function among them
    the same way. ``bump, peek = make_counter()`` remade by each Run All is
    the same ``peek`` while its ``n`` is; by identity, a statement reading
    it ran again each time. None when a value cannot be hashed."""
    # ``co_qualname`` is new in Python 3.11.
    parts = [held_code_digest(fn) or attribute_or(fn.__code__, "co_qualname", fn.__qualname__)]
    held: list[Any] = [*(fn.__defaults__ or ()), *(fn.__kwdefaults__ or {}).values()]
    for cell in fn.__closure__ or ():
        try:
            held.append(cell.cell_contents)
        except ValueError:  # an empty cell
            held.append(None)
    for value in held:
        if type(value) is types.FunctionType:
            part = _function_digest(value, depth + 1) if depth < 4 else None
        else:
            part = budgeted_hash(value)
            if part is not None and is_identity_fallback_hash(value, part):
                part = None
        if part is None:
            return None
        parts.append(part)
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def value_digests_key(source_hash: str, input_lineages: Mapping[str, str], occurrence_index: int = 0) -> str:
    """What the value digests of a statement's run are filed under
    (``TrackingState.no_cache_values``): its source, the lineages of its
    inputs and which copy of the statement in its cell it is, as the runtime
    and the simulation both build them. Two copies in one cell
    (``start = time.time()`` before and after the work) share source and
    inputs: filed together, the second's digest stood for the first's, and
    the work between them ran again for the reader below.

    Not its cache key: a key that cannot name what the code reaches carries
    a token of its own each time, one that reads a name no cell bound hashes
    its value (an iterator made before cash was on moves as it is read), and
    one that binds a name holding a module reads it as it was. Filed under
    such a key, the simulation did not find the digests, gave the output
    another lineage than the run did, and the statement ran a second time
    for each reader below: ``t = globals()[name](3)`` ran twice per Run All,
    and a draw in it drew twice."""
    parts = [
        source_hash,
        str(occurrence_index),
        *(f"{name}={lineage}" for name, lineage in sorted(input_lineages.items())),
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


#: Under this name in a statement's value digests (`no_cache_value_digest`):
#: the streams it drew from whose lineages its outputs' lineages leave out.
DRAWN_LEFT_OUT = "\x00drawn"


def without_drawn(input_lineages: Mapping[str, str], digests: Mapping[str, str] | None) -> list[str]:
    """The lineages of *input_lineages* an output keyed by its value takes:
    all but those of the streams the run drew from (`DRAWN_LEFT_OUT` in
    *digests*). ``t = next(it)`` moves ``it`` on every run; its value names
    what was drawn, so an equal draw keeps what is built on ``t`` served."""
    left_out = (digests or {}).get(DRAWN_LEFT_OUT)
    if not left_out:
        return list(input_lineages.values())
    names = set(left_out.split(","))
    return [lineage for name, lineage in input_lineages.items() if name not in names]


def no_cache_value_component(digests: Mapping[str, str] | None, name: str) -> str:
    """The lineage component for output *name* of a no-cache statement, from
    the digests its last run recorded; empty for any other statement."""
    if not digests or name not in digests:
        return ""
    return ":value:" + digests[name]


def key_hidden_reads(code: str, tracking_state: Any) -> set[str]:
    """Hidden variables the cache key of *code* reads: the RNG state a draw
    it spells consumes, plus any the runtime saw it draw from without
    spelling it (``model.fit()``), so a re-seed re-keys it."""
    return hidden_lineage_reads(code) | observed_rng_reads(tracking_state, code)


def rng_reads_now(tracking_state: Any, code: str) -> dict[str, str | None]:
    """The RNG variables the key of *code* reads that it draws from, each
    with its lineage now: None for an unseeded stream, which has none. Read
    before the statement's own draw moves the stream on, this is what its
    key read."""
    reads = key_hidden_reads(code, tracking_state)
    lineage = tracking_state.variable_lineage
    return {var: lineage.get(var) for var in drawn_rng_vars(reads, code, dict.fromkeys(reads, "read"))}


def note_rng_read(tracking_state: Any, code: str, read: Mapping[str, str | None]) -> None:
    """Record the lineages the RNG variables *code* drew from had when its
    key read them (*read*, None for an unseeded stream), for
    :func:`take_rng_reads_at_last_draw` and :func:`rng_reads_changed`."""
    try:
        digest = exact_source_digest(code)
    except (AttributeError, UnicodeEncodeError):
        return
    for ledger, kept in (
        (tracking_state.rng_read_at_last_draw, {var: lineage for var, lineage in read.items() if lineage is not None}),
        (tracking_state.rng_unseeded_at_last_draw, frozenset(var for var, lineage in read.items() if lineage is None)),
    ):
        if kept:
            ledger[digest] = kept
        else:
            ledger.pop(digest, None)


def take_rng_reads_at_last_draw(
    tracking_state: Any, code: str, reads: Iterable[str], virtual_lineage: dict[str, str]
) -> None:
    """Give each RNG variable among *reads* that *virtual_lineage* does not
    set yet the lineage it had when *code* last drew from it in this kernel.

    The simulation reads a stream no statement above set in its walk from
    the kernel, where it stands now: moved on by this very draw and any after
    it. With a stream a seed below the draw seeded on an earlier pass
    (``np.random.seed(0)`` under ``gs.fit(X, y)``), the check before the next
    cell keyed the fit on a stream the fit never read, missed the entry the
    run had just stored, and ran the fit again. The stream as the draw read
    it is the one the value in memory came from.
    """
    if not code:
        return
    try:
        digest = exact_source_digest(code)
    except (AttributeError, UnicodeEncodeError):
        return
    read = (attribute_or(tracking_state, "rng_read_at_last_draw", None) or {}).get(digest) or {}
    unseeded = (attribute_or(tracking_state, "rng_unseeded_at_last_draw", None) or {}).get(digest) or ()
    for var in reads:
        if var in virtual_lineage:
            continue
        if var in read:
            virtual_lineage[var] = read[var]
        elif var in unseeded:
            # Unseeded where it drew: no lineage (`UNSEEDED`), not the one a
            # seed the draw set itself, or one after it, gave the stream
            # since (``random.seed`` in a helper the statement called).
            virtual_lineage[var] = UNSEEDED


#: What the simulation's lineage holds for an RNG variable a statement
#: read unseeded (`take_rng_reads_at_last_draw`): no lineage, as the
#: runtime's had, which keys no input and moves on no draw
#: (``drawn_rng_vars``).
UNSEEDED = ""


def rng_reads_changed(tracking_state: Any, code: str) -> list[str]:
    """The RNG variables *code* last drew from in this kernel whose lineage
    is not the one that draw read: the random state its key read changed.

    A stream it drew from unseeded counts when a seed has given it a
    lineage since: ``gs.fit(X, y)`` above ``np.random.seed(0)`` draws
    unseeded on a first Run All and from the seeded stream on the next one
    when the cells above it are not known. Read before the statement's own
    draw moves the stream on (``_advance_rng``), as the miss reason is.
    Empty on a first run.
    """
    if not code:
        return []
    try:
        digest = exact_source_digest(code)
    except (AttributeError, UnicodeEncodeError):
        return []
    current = tracking_state.variable_lineage
    read = (attribute_or(tracking_state, "rng_read_at_last_draw", None) or {}).get(digest) or {}
    unseeded = (attribute_or(tracking_state, "rng_unseeded_at_last_draw", None) or {}).get(digest) or ()
    changed = {var for var, was in read.items() if current.get(var) != was}
    changed |= {var for var in unseeded if current.get(var) is not None}
    return sorted(changed)


def note_seeded_unseen(tracking_state: Any, code: str, key: str, adopted: Mapping[str, str]) -> None:
    """Record that the statement *code*, keyed *key*, left the RNG variables
    of *adopted* at those lineages through a seed no statement spells (one
    set inside a function it called), for :func:`seeded_unseen_after`."""
    if not code or not key or not adopted:
        return
    try:
        digest = exact_source_digest(code)
    except (AttributeError, UnicodeEncodeError):
        return
    tracking_state.rng_seeded_unseen_by.setdefault(digest, {})[key] = dict(adopted)


def seeded_unseen_by(tracking_state: Any, code: str) -> Mapping[str, Mapping[str, str]] | None:
    """``{key: {RNG variable: lineage}}`` the runtime recorded for *code*
    (:func:`note_seeded_unseen`), or None: cheap, so the simulation builds
    the statement's key only for one that has some."""
    ledger = attribute_or(tracking_state, "rng_seeded_unseen_by", None)
    if not ledger or not code:
        return None
    try:
        return ledger.get(exact_source_digest(code))
    except (AttributeError, UnicodeEncodeError):
        return None


def lineage_hidden_reads(code: str) -> set[str]:
    """Hidden variables the output lineage of *code* reads: only the draws it
    spells. An observed draw stays out: folding it in makes a refit mint a
    new lineage on every run, and reconstruction then never converges."""
    return hidden_lineage_reads(code)


def body_only_reads(code: str) -> frozenset[str]:
    """Names the ``def`` statement *code* reads only in its body, which runs
    at call time; empty for any other statement.

    Only for one undecorated function: a decorator can keep the function
    where no call site walks into its code (an ``lru_cache`` wrapper has
    none). What the ``def`` itself evaluates -- its defaults -- is not here.
    """
    tree = parse_cached(code)
    if tree is None or len(tree.body) != 1:
        return frozenset()
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.decorator_list:
        return frozenset()
    evaluated = [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]
    at_def = {n.id for d in evaluated for n in walk_tree(d) if isinstance(n, ast.Name)}
    in_body = {n.id for stmt in node.body for n in walk_tree(stmt) if isinstance(n, ast.Name)}
    return frozenset(in_body - at_def)


def call_time_reads(code: str, user_ns: Mapping[str, Any], virtual_modules: Iterable[str] = ()) -> frozenset[str]:
    """The `body_only_reads` of *code* that a function's identity
    (`function_identity`) leaves out: all but the modules. A key reading
    the function folds the others as they are at the call; a module only
    through the function's lineage.

    A module by what it is, as `cache_key.called_function_dependencies`
    tells it, not by `is_module_like`: that also takes a callable named
    ``_x`` for one, and the simulation that keys a restart sees no value for
    a function defined in the same cell (``def _ran`` above ``def load``),
    so it noted another identity than the run before and the cell's final
    value was rebuilt instead of restored."""
    modules = set(virtual_modules)
    return frozenset(
        name
        for name in body_only_reads(code)
        if name not in modules and not instance_of(user_ns.get(name), types.ModuleType)
    )


#: A ``def``'s source digest -> the globals its body named that were not
#: bound when it ran: helpers defined below it (`def_time_names`).
_FORWARD_HELPERS: LruMemo[str, frozenset[str]] = LruMemo(CODE_OBJECTS)
#: The names in any entry of `_FORWARD_HELPERS`, for `forward_helper`.
_FORWARD_NAMES: LruMemo[str, bool] = LruMemo(CODE_OBJECTS)


def def_time_names(
    code: str,
    names: Iterable[str],
    user_ns: Mapping[str, Any],
    virtual_modules: Iterable[str] = (),
    bound: Iterable[str] = (),
) -> set[str]:
    """*names* less the helpers defined below the ``def`` statement *code*:
    the names whose lineages and sources its function's lineage folds.

    A helper only the body calls is looked up when the function is called,
    not when the ``def`` runs, and a key reading the function folds it there
    (``cache_key.called_function_dependencies``). Folded into the lineage as
    it stood when the ``def`` ran, a helper defined below gave the function
    another lineage on each Run All: absent on the first, the earlier run's
    function on the next. What holds the function took those lineages, so a
    cell reading ``exact = (greedy, held_karp)`` missed on the next Run All,
    and the upstream check re-ran the ``def`` as broken.

    Only such a helper is left out: a name the body only calls that was not
    bound when the ``def`` ran -- in the kernel, or for the simulation in
    *bound*, what it bound above -- and one bound since to a function, plain
    or wrapped, where this ``def`` had found nothing. Every other global
    keeps folding as it stood at the ``def``, so the function's lineage
    moves when one changed. Both engines take the names from here."""
    return set(names) - forward_helpers(code, user_ns, virtual_modules, bound)


def forward_helpers(
    code: str,
    user_ns: Mapping[str, Any],
    virtual_modules: Iterable[str] = (),
    bound: Iterable[str] = (),
) -> frozenset[str]:
    """The helpers defined below the ``def`` statement *code* that
    `def_time_names` leaves out: what its body calls that was not bound
    when it ran, or is bound since to a function where it had found
    nothing. Empty for any other statement."""
    called = _called_helpers(code) & call_time_reads(code, user_ns, virtual_modules)
    if not called:
        return frozenset()
    # Asked by name: a whole copy of the simulation's lineage map was a
    # read of every name in it (``cell_memo``).
    if not isinstance(bound, (Mapping, set, frozenset)):
        bound = set(bound)
    digest = exact_source_digest(code)
    seen = _FORWARD_HELPERS.get(digest) or frozenset()
    absent = frozenset(name for name in called if name not in user_ns and name not in bound)
    if absent - seen:
        seen = seen | absent
        _FORWARD_HELPERS[digest] = seen
        for name in absent:
            _FORWARD_NAMES[name] = True
    return absent | {name for name in called & seen if _is_helper(user_ns.get(name))}


def _called_helpers(code: str) -> frozenset[str]:
    """Globals the body of the ``def`` statement *code* calls by name,
    builtins left out: the helpers it may name before they exist."""
    tree = parse_cached(code)
    if tree is None or len(tree.body) != 1:
        return frozenset()
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return frozenset()
    called = {
        sub.func.id
        for stmt in node.body
        for sub in walk_tree(stmt)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
    }
    return frozenset(name for name in called if name not in vars(builtins))


def _is_helper(value: Any) -> bool:
    """A function, plain or wrapped (``functools.cache``): what a ``def``
    below binds. A class is left out: the ``def`` reads its code."""
    return callable(value) and not isinstance(value, type)


def forward_helper(name: str, user_ns: Mapping[str, Any]) -> bool:
    """Whether *name* may be a helper `def_time_names` leaves out of a
    function's lineage: a name some ``def`` read before it was bound, and
    unbound or a function now. A key folds it at the call instead."""
    if name not in _FORWARD_NAMES:
        return False
    return name not in user_ns or _is_helper(user_ns.get(name))


#: A notebook function's lineage -> its identity (`function_identity`), for
#: this process. One entry per ``def`` run; a dropped one only re-keys.
_FUNCTION_IDENTITIES: LruMemo[str, str] = LruMemo(CODE_OBJECTS)


def function_identity(lineage: str) -> str:
    """What *lineage* contributes to a cache key (``cache_key``): for a
    function a plain ``def`` bound, the lineage of its code and of what the
    ``def`` evaluated, without the globals its body reads; for any other
    lineage, itself.

    The function's lineage folds the globals its body reads as they were
    when the ``def`` last ran. That is what moves it on when the ``def``
    runs again after one changed, so the upstream check re-runs what holds
    the function, and lineages keep it. In a key it cost the reuse a restart
    should give: ``num_cols`` edited without re-running ``def make_prep()``,
    and after a restart the ``def`` ran with the new ``num_cols``, so every
    fit calling it missed with its code and data unchanged. A key reading
    the function also folds those globals as they are now
    (``cache_key.called_function_dependencies``), so the identity leaves
    them out and the key loses nothing.
    """
    return _FUNCTION_IDENTITIES.get(lineage) or lineage


def note_function_identity(
    lineage: str,
    code: str,
    source_hash: str,
    def_time_lineages: Iterable[str],
    func_component: str,
    user_ns: Mapping[str, Any] | None,
) -> None:
    """Record the identity (`function_identity`) of the function the ``def``
    statement *code* bound with *lineage*. *def_time_lineages* are the input
    lineages less those of `call_time_reads`. Both engines call this, with
    the same ingredients, wherever they give a ``def`` its lineage."""
    identity = output_lineage(
        source_hash,
        def_time_lineages,
        "",
        func_component,
        "",
        ":defn" + statement_environment_component(code, user_ns),
    )
    if identity != lineage:
        _FUNCTION_IDENTITIES[lineage] = identity


def input_lineage(
    name: str,
    namespace: Mapping[str, Any],
    lineages: Iterable[Mapping[str, str]],
    *,
    compute_hash: Callable[[Any], str] | None,
    function_tracker: Any,
    code: str | None,
    virtual_modules: Iterable[str] = (),
) -> str | None:
    """What input *name* contributes to the output lineage of the statement
    *code*, or None for nothing.

    A module read by plain attribute access is valued by what those
    attributes reach (:func:`module_read_lineage`). Otherwise the first of
    *lineages* that knows *name* wins; the runtime passes its recorded
    lineages, the simulation its own in front of them. A name with neither is
    valued by its content, unless it is None, cash's instrumentation or
    module-like (:func:`is_module_like`), which contribute nothing, as in the
    cache key. *compute_hash* None hashes ``str(value)``, as the key does.
    """
    value = namespace.get(name)
    narrowed = module_read_lineage(function_tracker, name, value, code)
    if narrowed is not None:
        return narrowed
    for known in lineages:
        if name in known:
            return known[name]
    if value is None or is_cash_instrumentation(value) or is_module_like(name, value, virtual_modules):
        return None
    try:
        if compute_hash is None:
            return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        return compute_hash(value)
    except (TypeError, ValueError, AttributeError, pickle.PicklingError) as exc:
        logger.debug("Could not hash input %r for its lineage: %s", name, exc)
        return None


def statement_environment_reads(
    code: str, user_ns: Mapping[str, Any] | None = None, reach: Reach | None = None
) -> set[tuple[str, str]]:
    """The environment reads whose value a key can fold
    (`cash.effects.environment_input`): ``os.getenv("NAME")``,
    ``os.environ["NAME"]``, ``os.getcwd()``, ``Path.cwd()``, a relative path
    made absolute (``os.path.abspath(p)``, ``Path(p).resolve()``), written in *code* or in a
    function of the user's it calls (:func:`callee_environment_reads`;
    *reach* as there).
    """
    return _written_environment_reads(code, user_ns) | callee_environment_reads(code, user_ns, reach)


@functools.lru_cache(maxsize=NOTEBOOK_STATEMENTS)
def _names_the_environment(code: str) -> bool:
    """Whether *code*'s text holds an `ENVIRONMENT_TEXT_MARKERS` marker,
    once per text: the upstream scan asks it of every cell above, on every
    check."""
    return any(marker in code for marker in ENVIRONMENT_TEXT_MARKERS)


def _written_environment_reads(code: str, user_ns: Mapping[str, Any] | None) -> set[tuple[str, str]]:
    """The environment reads written in *code* itself."""
    if not code or not _names_the_environment(code):
        return set()
    tree = parse_cached(code)
    if tree is None:
        return set()
    found = set()
    for node in walk_tree(tree):
        entry = environment_input(node, user_ns)
        if entry is not None:
            found.add(entry)
    return found


def callee_environment_reads(
    code: str, user_ns: Mapping[str, Any] | None, reach: Reach | None = None
) -> set[tuple[str, str]]:
    """The environment reads inside the user's functions *code* calls.

    ``m = mylib.mode()`` with ``mode`` reading ``os.environ["MODE"]``: the
    statement's text reads no environment, and the cell setting ``MODE`` was
    edited and the notebook run again, the old mode was served. The functions
    are found by :func:`~cash.notebook.callee_reach.reached_user_code` (or
    given as *reach*) and read by the decorator's own analysis
    (`PurityReport.environment_reads`), which follows the helpers they call,
    in their module and in others of the user's.
    """
    if not user_ns:
        return set()
    if reach is None:
        reach = reached_user_code(code, user_ns)
    found: set[tuple[str, str]] = set()
    covered: set[int] = set()
    for fn in reach.functions:
        if id(fn) in covered:
            continue  # its reads are in a caller's report already
        try:
            report = get_analyzer().analyze(fn)
        except Exception:
            logger.debug("Could not analyse %r for the environment it reads", fn, exc_info=True)
            continue
        found |= report.environment_reads
        covered |= report.covered_ids()
    return found


#: Clocks that only measure time passing: a busy-wait, a timeout loop or a
#: measured duration reads them, and a value built from one differs between
#: two plain runs too. Left out of `helper_ambient_reads`.
_DURATION_CLOCKS = tuple(
    f"time.{name}("
    for name in (
        "perf_counter",
        "perf_counter_ns",
        "monotonic",
        "monotonic_ns",
        "process_time",
        "process_time_ns",
        "thread_time",
        "thread_time_ns",
    )
)


def helper_ambient_reads(
    code: str, user_ns: Mapping[str, Any] | None, reach: Reach | None = None
) -> list[tuple[Any, Any]]:
    """``(function called, ambient-read issue)`` for each clock, UUID or
    random-bytes read inside the user's functions *code* calls.

    ``s = stamp(1)`` with ``stamp`` doing work and returning ``time.time()``:
    the statement's text reads no clock, so it was cached and the first time
    came back on every run. The functions are those
    :func:`~cash.notebook.callee_reach.reached_user_code` finds, read by the
    decorator's own analysis (``ISSUE_AMBIENT_READ``), which follows the
    helpers they call and leaves out a read that only reaches a log line. A
    ``@cash.cache`` function is left to its own warning.
    """
    if not user_ns or "(" not in code:
        return []
    from ..analysis.purity_report import ISSUE_AMBIENT_READ
    from ..decorator.user_code import is_cash_wrapper

    if reach is None:
        reach = reached_user_code(code, user_ns)
    # One entry per read, named by the function that makes it: the analysis
    # of a caller follows its helpers, so ``label`` calling ``stamp`` finds
    # ``stamp``'s read too, and the order of *reach* decided which came first.
    found: dict[tuple[str, str], tuple[Any, Any]] = {}
    # A function a caller's walk audited whole (`PurityReport.covered_ids`)
    # finds nothing that caller did not: only its own reads are named for it.
    covered: dict[int, tuple[Any, ...]] = {}
    for fn in reach.functions:
        if is_cash_wrapper(fn):
            continue
        if id(fn) in covered:
            for issue in covered[id(fn)]:
                if issue.kind == ISSUE_AMBIENT_READ and _makes_the_read(fn, issue.where):
                    mark = (issue.where, issue.description)
                    if mark in found:
                        found[mark] = (fn, issue)
            continue
        try:
            report = get_analyzer().analyze(fn)
        except Exception:
            logger.debug("Could not analyse %r for the ambient state it reads", fn, exc_info=True)
            continue
        issues = report.issues
        covered.update(dict.fromkeys(report.covered_ids(), issues))
        for issue in issues:
            mark = (issue.where, issue.description)
            if issue.kind != ISSUE_AMBIENT_READ or issue.description.startswith(_DURATION_CLOCKS):
                continue
            if mark not in found or _makes_the_read(fn, issue.where):
                found[mark] = (fn, issue)
    return list(found.values())


def _makes_the_read(fn: Any, where: str) -> bool:
    """Is *fn* the function the analysis says reads (``where`` is
    ``module.qualname``)? By both: a ``stamp`` of the notebook is not
    ``Clock.stamp``."""
    module, qualname = attribute_or(fn, "__module__", None), attribute_or(fn, "__qualname__", None)
    return isinstance(qualname, str) and where in (f"{module}.{qualname}", qualname)


def module_data_component(reach: Reach) -> str:
    """What the data of the user's modules that *reach* reads holds now,
    digested: empty when it reads none.

    ``b = mylib.from_k(10)`` with ``from_k`` reading the module's ``K``: a
    cell setting ``mylib.K = 7`` (or ``mylib.CONFIG["k"] = 7``,
    ``mylib.set_k(7)``, ``setattr(mylib, "K", 7)``) changes the answer and
    leaves the module's file as it was, so a key built from the file served
    the old answer. The values are folded, as the decorator folds the module
    globals a cached function reads, however they were set.
    """
    return module_data_parts_component(module_data_digests(reach))


#: How a module value that cannot be hashed starts its digest: one of its
#: own each time, so it never matches.
UNHASHABLE = "unhashable:"


def module_data_digest(label: str, value: Any) -> str:
    """The digest of *value*, the module data at *label*: its value hash,
    with the code of the functions and of the user's objects it holds
    (`held_code_digest`). The value hash pickles a function by its name, so
    a registry holding a function edited since keyed as it did.

    A stream or random generator set on a library's module
    (`library_draw_source`) is keyed by which object it is: each read draws
    from it, so its state after a draw differs from the state the draw was
    keyed on, and a cell reading what the draw bound found it moved on and
    drew again to rebuild it. The cell setting it makes a new one."""
    from .callee_reach import library_draw_source

    try:
        if library_draw_source(label, value):
            return f"drawn:{type(value).__qualname__}:{id(value)}"
        digest = compute_hash(value)
        code = held_code_digest(value)
        if code:
            digest = hashlib.sha256(f"{digest}:code:{code}".encode("utf-8")).hexdigest()
        return digest
    except Exception:
        logger.debug("Could not hash module data %s", label, exc_info=True)
        return UNHASHABLE + secrets.token_hex(16)


#: How deep `held_code_at_digest` follows routines held by routines (a
#: wrapper's closure holding the function it wraps, a method's function).
_CODE_AT_DEPTH = 8


def held_code_at_digest(value: Any) -> str:
    """Which routine *value*, found at a module's attribute, is: everything
    its answer turns on, digested the same in every process.

    A function: the code it holds (`held_code_digest`), its own compiled
    body and constants (two lambdas on one line have one source), and the
    data its defaults and closure hold (``mk(1)`` and ``mk(2)``, a default
    ``Random(1)`` or ``Random(2)``). A bound method: its function and the
    content of the object it is bound to (``C(1).get``, ``{'a': 1}.get``,
    ``'aaa'.count``). Another routine (``math.sqrt``, ``np.sum``): its type
    and qualified name.

    `UNHASHABLE` when a part cannot be digested: the key then matches no
    entry (`compute_cache_key`)."""
    try:
        parts: list[str] = []
        _routine_parts(value, parts, 0, {})
    except Exception:
        logger.debug("Could not digest the routine at a module attribute", exc_info=True)
        return UNHASHABLE + secrets.token_hex(16)
    if any(part.startswith(UNHASHABLE) for part in parts):
        return UNHASHABLE + secrets.token_hex(16)
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _routine_parts(value: Any, parts: list[str], depth: int, met: dict[int, int]) -> None:
    """Add to *parts* what *value*'s answer turns on (`held_code_at_digest`).
    A routine met before on the way (a recursive inner function its own
    closure holds) is named by the order it was first met in."""
    if id(value) in met:
        parts.append(f"again:{met[id(value)]}")
        return
    if depth > _CODE_AT_DEPTH:
        parts.append(UNHASHABLE)
        return
    met[id(value)] = len(met)
    kind = type(value)
    if kind is types.FunctionType:
        parts.append("fn:" + held_code_digest(value))
        parts.append("body:" + _body_digest(value.__code__))
        held: list[Any] = [*(value.__defaults__ or ()), *sorted((value.__kwdefaults__ or {}).items())]
        for cell in value.__closure__ or ():
            try:
                held.append(cell.cell_contents)
            except ValueError:  # an empty cell
                held.append(None)
        for item in held:
            _held_part(item, parts, depth, met)
        return
    owner = attribute_or(value, "__self__", None)
    if kind is types.MethodType:
        _routine_parts(value.__func__, parts, depth + 1, met)
        _held_part(owner, parts, depth, met)
        return
    name = f"{attribute_or(value, '__module__', None)}.{attribute_or(value, '__qualname__', None)}"
    parts.append(f"named:{kind.__module__}.{kind.__qualname__}:{name}")
    if owner is not None and not instance_of(owner, types.ModuleType):
        # A builtin bound to an object: ``{'a': 1}.get``, ``'aaa'.count``.
        _held_part(owner, parts, depth, met)


def _held_part(item: Any, parts: list[str], depth: int, met: dict[int, int]) -> None:
    """A value a routine holds: a function or method by `_routine_parts`, a
    class or module by its name, anything else by its content and the code
    it holds, as `module_data_digest` keys module data; one hashed only by
    which object it is, as none."""
    if item is None or instance_of(item, types.ModuleType):
        parts.append(f"held:{attribute_or(item, '__name__', None)}")
    elif instance_of(item, type):
        parts.append(f"class:{attribute_or(item, '__module__', None)}.{attribute_or(item, '__qualname__', None)}")
    elif type(item) in (types.FunctionType, types.MethodType) or (
        is_routine(item) and attribute_or(item, "__self__", None) is not None
    ):
        _routine_parts(item, parts, depth + 1, met)
    else:
        digest = compute_hash(item)
        if is_identity_fallback_hash(item, digest):
            # Which object, not what it holds: blind to a change in place,
            # and another in the next process.
            parts.append(UNHASHABLE)
            return
        code = held_code_digest(item)
        parts.append(f"data:{digest}:{code}")


def _body_digest(code: types.CodeType) -> str:
    """*code*'s compiled body, constants and names, nested code included:
    the same in every process, unlike its file name and positions."""
    found: list[str] = []
    stack = [code]
    while stack:
        co = stack.pop()
        found.append(co.co_code.hex())
        found.append(repr(co.co_names))
        for const in co.co_consts:
            if isinstance(const, types.CodeType):
                stack.append(const)
            elif isinstance(const, frozenset):
                # ``x in {"a", "b"}``: its order follows the string hashes.
                found.append(repr(sorted(map(repr, const))))
            else:
                found.append(repr(const))
    return hashlib.sha256("\x00".join(found).encode("utf-8")).hexdigest()


def held_code_component(reach: Reach) -> str:
    """The key component of the routines *reach* found at modules'
    attributes (`Reach.code_at`): empty when it found none."""
    parts = []
    for label, value in reach.code_at:
        digest = held_code_at_digest(value)
        if digest.startswith(UNHASHABLE):
            _warn_unread_routine(label, value)
        parts.append(f"{label}={digest}")
    joined = ":".join(parts)
    return f":codeat:{joined}" if joined else ""


def unread_routine_in(component: str) -> bool:
    """Whether *component* (`held_code_component`'s) has a routine that
    could not be digested: a key holding it must match no entry."""
    return f"={UNHASHABLE}" in component.partition(":codeat:")[2]


#: The labels `_warn_unread_routine` has warned about, in this process.
_UNREAD_ROUTINES_WARNED: set[str] = set()


def _warn_unread_routine(label: str, value: Any) -> None:
    """Say once that a call reading the routine at *label* runs uncached."""
    if label in _UNREAD_ROUTINES_WARNED:
        return
    _UNREAD_ROUTINES_WARNED.add(label)
    from ..diagnostics import warn_diagnostic
    from ..exceptions import CashCacheIneffectiveWarning

    warn_diagnostic(
        CashCacheIneffectiveWarning,
        "KEY-UNHASHABLE-GLOBAL",
        f"a call reads '{label.lstrip('@')}', a {type(value).__qualname__} holding a value cash could not hash, "
        "so it cannot tell that routine from another put there: the call runs every time.",
        "register a hasher for the type it holds with cash.register_hasher, or set a function that "
        "holds plain values there.",
    )


def module_data_digests(reach: Reach) -> list[tuple[str, str]]:
    """``(label, digest)`` of each piece of module data *reach* reads, in its order."""
    return [(label, module_data_digest(label, value)) for label, value in reach.data]


def module_data_parts_component(parts: Iterable[tuple[str, str]]) -> str:
    """The key component of the ``(label, digest)`` *parts*."""
    joined = ":".join(f"{label}={digest}" for label, digest in parts)
    return f":moddata:{joined}" if joined else ""


def statement_environment_component(code: str, user_ns: Mapping[str, Any] | None = None) -> str:
    """What the environment reads in *code* return now, digested: empty when
    it reads none.

    ONE function for the cache key (``cache_key``), the output lineage the
    runtime records (``statement/lineage.py``) and the one the simulation
    recomputes (``upstream/virtual_lineage.py``). The key alone was not
    enough: ``t = os.getenv("TENANT")`` re-ran for a new tenant, but ``t``
    kept its lineage, so ``u = t.upper()`` below it hit and returned the
    first tenant's answer.
    """
    reads = statement_environment_reads(code, user_ns)
    return environment_component(reads) if reads else ""


def statement_module_data(code: str, user_ns: Mapping[str, Any] | None) -> str:
    """:func:`module_data_component` of what *code* reaches now."""
    if not code or not user_ns:
        return ""
    return module_data_component(reached_user_code(code, user_ns))


def recorded_reads_lineage_component(
    tracking_state: Any,
    cache_key: str | None,
    code: str,
    user_ns: Mapping[str, Any] | None,
    outputs: Iterable[str] = (),
    *,
    as_last_run: bool = False,
    simulated: Mapping[str, str] | None = None,
) -> str:
    """The environment and the module data the statement keyed *cache_key*
    read, and the globals its called functions reached for, for its
    outputs' lineage: what its key folded (``TrackingState.reads``), so a
    reader below misses when a value changed. A key nothing recorded reads
    the environment now. ONE
    lookup for the runtime (``statement/lineage.py``) and the simulation
    (``upstream/statement_lineage.py``). The simulation passes
    *as_last_run*: the callee globals are then the ones the statement's last
    run was keyed with, when it ran in this kernel, but for the helpers
    defined below a function that it bound itself above the statement
    (*simulated*), which it keys as the runtime finds them there: one
    edited above and not run yet. The function's lineage does not carry
    them (`def_time_names`).
    """
    reads = attribute_or(tracking_state, "reads", None)
    by_key = attribute_or(reads, "by_key", None)
    if by_key and cache_key and cache_key in by_key:
        environment = by_key[cache_key]
    else:
        environment = statement_environment_component(code, user_ns)
    callees = attribute_or(reads, "callees", None)
    by_code = attribute_or(reads, "callees_by_code", None)
    deps = None
    if as_last_run and by_code and code in by_code:
        deps = by_code[code]
        own = callees.get(cache_key) if callees and cache_key else None
        if simulated and own:
            deps = _edited_since(
                deps, own, simulated, attribute_or(tracking_state, "variable_lineage", None) or {}, user_ns
            )
    elif callees and cache_key and cache_key in callees:
        deps = callees[cache_key]
    if deps:
        # Less the globals the statement itself changes (``p = load()`` with
        # ``load`` appending to ``CALLS``): they are its outputs, and the
        # simulation reads them where the run left them, so its key differs
        # from the runtime's there.
        written = set(outputs)
        kept = [dep for dep in deps if dep.split(":", 1)[0] not in written]
        if kept:
            environment += ":callees:" + hashlib.sha256(":".join(kept).encode("utf-8")).hexdigest()
    return environment


def _edited_since(
    last_run: tuple[str, ...],
    own: Iterable[str],
    simulated: Mapping[str, str],
    live: Mapping[str, str],
    user_ns: Mapping[str, Any],
) -> tuple[str, ...]:
    """*last_run*'s callee globals, but for the helpers defined below a
    function (`forward_helper`) the simulation gave another lineage than the
    kernel holds while the kernel holds what the last run read: one edited
    above and not run yet. Any other global keeps the last run's."""
    changed = {}
    for dep in own:
        name = dep.split(":", 1)[0]
        if not forward_helper(name, user_ns):
            continue
        if name in simulated and name in live and simulated[name] != live[name]:
            changed[name] = dep
    kept = []
    for dep in last_run:
        name, _, lineage = dep.partition(":")
        if name in changed and name in live and lineage == function_identity(live[name]):
            kept.append(changed[name])
        else:
            kept.append(dep)
    return tuple(kept)


def callable_source_component(function_tracker: Any, inputs: set[str], user_ns: dict) -> str:
    """``:name:hash`` for every user-defined callable among *inputs*."""
    if function_tracker is None:
        return ""
    hashes = function_tracker.get_callable_source_hashes(inputs, user_ns)
    if not hashes:
        return ""
    return ":" + ":".join(f"{k}:{v}" for k, v in sorted(hashes.items()))


def _tracked(function_tracker: Any) -> set[str]:
    return attribute_or(function_tracker, "tracked_modules", None) or set()


def _closure_with_deps(
    function_tracker: Any,
    mod_file: str,
    owners: set[str],
    attrs: Iterable[str],
    tag: str,
) -> str | None:
    """What *attrs* reach inside *mod_file*, plus the module's tracked
    dependency files whole; None when the closure cannot be bounded.

    Shared by the two narrowings -- a module read by attribute
    (:func:`module_read_lineage`) and a name brought in by ``from ... import``
    (:func:`_from_module_hash`) -- so they bound a closure the same way.
    """

    digest = closure_digest(mod_file, attrs)
    if digest is None:
        return None
    parents = attribute_or(function_tracker, "dep_file_to_parents", None) or {}
    dep_files = sorted(dep for dep, o in parents.items() if owners & set(o))
    h = hashlib.sha256(("%s:%s" % (tag, digest)).encode("utf-8"))
    for dep in dep_files:
        h.update((":" + (read_module_source_hash(dep) or "missing")).encode("utf-8"))
    return h.hexdigest()


def _from_module_hash(module_name: str, function_tracker: Any, name: str | None = None) -> str:
    """The source component of a name imported from *module_name*.

    With the *name* the import statement read, only what that name reaches
    inside the module: `from helpers import load` then depends on `load`, and
    editing `report` in the same file no longer re-runs everything built on
    `load`.
    Without one, or when the closure cannot be
    bounded, the whole module, as before.
    """
    if module_name not in _tracked(function_tracker):
        return ""
    mod_obj = sys.modules.get(module_name)
    mod_file = attribute_or(mod_obj, "__file__", None) if mod_obj else None
    if not (mod_file and os.path.isfile(mod_file)):
        return ""
    if name is not None:
        narrowed = _closure_with_deps(function_tracker, mod_file, {module_name}, {name}, "fromsym:" + module_name)
        if narrowed is not None:
            return f":from_sym_src:{narrowed}"
    digest = read_module_source_hash(mod_file)
    return f":from_mod_src:{digest}" if digest else ""


def imported_from(var_name: str, code: str, tree: ast.Module | None = None) -> tuple[str, str] | None:
    """``(module, name)`` when a ``from ... import`` in *code* bound *var_name*."""
    try:
        parsed = tree if tree is not None else ast.parse(code.strip())
    except SyntaxError:
        return None
    for node in parsed.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if (alias.asname or alias.name) == var_name and alias.name != "*":
                    return node.module, alias.name
    return None


def _imported_name(var_name: str, code: str, tree: ast.Module | None) -> str | None:
    """The name a ``from ... import`` in *code* bound as *var_name*, if one did."""
    found = imported_from(var_name, code, tree)
    return found[1] if found else None


def module_source_component(
    function_tracker: Any,
    value: Any,
    var_name: str,
    code: str,
    tree: ast.Module | None = None,
    note_from_import: Callable[[str, str], None] | None = None,
) -> str:
    """The source hash of the local module *var_name* came from, or ``""``.

    Three cases: ``import X`` of a tracked module (its source plus its tracked
    dependency files); a callable from a tracked module
    (``from X import func``); a non-callable imported from one
    (``from X import CONST``, found through the import statement itself).
    *note_from_import* is told ``(var_name, module)`` whenever *var_name* comes
    from another module through a ``from`` import -- the runtime keeps that map
    for module invalidation.
    """
    if function_tracker is None:
        return ""
    if instance_of(value, types.ModuleType):
        mod_file = attribute_or(value, "__file__", None)
        # By the module's OWN name, not the name the cell bound it to.
        # `tracked_modules` holds real module names, so `import tickets_lib
        # as tl` failed this test and returned "" -- the module's source
        # stayed out of `tl`'s lineage, `tl` keyed identically before and
        # after an edit to tickets_lib.py, and every statement built on it
        # kept its cached value. Reloading worked, the badge said MODULE
        # RELOADED, and the cell still returned the pre-edit answer until the
        # kernel was restarted.
        #
        # `var_name` is still accepted so a tracker that registered the bound
        # name keeps working; for an unaliased import the two are equal, so
        # no existing lineage moves.
        names = {attribute_or(value, "__name__", var_name), var_name}
        if not (mod_file and os.path.isfile(mod_file) and names & _tracked(function_tracker)):
            return ""
        parents = attribute_or(function_tracker, "dep_file_to_parents", None) or {}
        dep_files = {dep for dep, owners in parents.items() if names & set(owners)}
        digest = read_module_source_hash(mod_file, dep_files)
        return f":mod_src:{digest}" if digest else ""

    if callable(value):
        obj_module = attribute_or(value, "__module__", None)
        if not (obj_module and obj_module in _tracked(function_tracker)):
            return ""
        if note_from_import is not None:
            note_from_import(var_name, obj_module)
        # Narrowed only when THIS statement is the `from ... import` that bound
        # it. A callable a module function returns (`fn = helpers.make()`)
        # has no import to read a name from, and keeps the whole module.
        return _from_module_hash(obj_module, function_tracker, _imported_name(var_name, code, tree))

    try:
        parsed = tree if tree is not None else ast.parse(code.strip())
    except SyntaxError:
        return ""
    for node in parsed.body:
        if not (isinstance(node, ast.ImportFrom) and node.module):
            continue
        for alias in node.names:
            if (alias.asname or alias.name) == var_name:
                if note_from_import is not None:
                    note_from_import(var_name, node.module)
                return _from_module_hash(node.module, function_tracker, alias.name)
    return ""

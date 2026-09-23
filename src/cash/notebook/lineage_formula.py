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
import hashlib
import logging
import os
import pickle
import sys
import types
from collections.abc import Mapping
from typing import Any, Callable, Iterable

from ..effects import environment_component, environment_input
from ..tracking.module_symbols import closure_digest, static_attribute_reads
from ..tracking.randomness import hidden_lineage_reads, observed_rng_reads

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
    if getattr(val, "_is_file_tracker_patch", False) is True:
        return True
    shell = sys.modules.get("IPython.core.interactiveshell")
    return shell is not None and val is getattr(shell, "_modified_open", _NO_SHELL_OPEN)


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
        if isinstance(val, types.ModuleType):
            return True
        # IPython internals and bound methods behave like modules -- skip them
        if callable(val) and (var_name.startswith("_") or hasattr(val, "__self__")):
            return True
    except (AttributeError, TypeError) as exc:
        logger.debug("[CACHE_KEY] Failed to check module/callable type for '%s': %s", var_name, exc)
    return False


def read_module_source_hash(mod_file: str, dep_files: set[str] | None = None) -> str | None:
    # Imported on use: the ``statement`` package imports this module.
    # Local: import cycle lineage_formula -> statement.file_deps -> ... -> lineage_formula.
    from .statement.file_deps import read_module_source_hash as read

    return read(mod_file, dep_files)


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
    it. Round 27, r27s2: editing one helper re-read all 10,000 of their
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
    if function_tracker is None or not code or not isinstance(value, types.ModuleType):
        return None
    mod_file = getattr(value, "__file__", None)
    real = getattr(value, "__name__", var_name)
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
) -> str:
    """The lineage hash of one output of a statement."""
    lineage_str = (
        f"{source_hash}:{':'.join(sorted(input_lineages))}{file_component}{func_component}{module_component}"
        f"{environment}"
    )
    return hashlib.sha256(lineage_str.encode("utf-8")).hexdigest()


def key_hidden_reads(code: str, tracking_state: Any) -> set[str]:
    """Hidden variables the cache key of *code* reads: the RNG state a draw
    it spells consumes, plus any the runtime saw it draw from without
    spelling it (``model.fit()``), so a re-seed re-keys it."""
    return hidden_lineage_reads(code) | observed_rng_reads(tracking_state, code)


def lineage_hidden_reads(code: str) -> set[str]:
    """Hidden variables the output lineage of *code* reads: only the draws it
    spells. An observed draw stays out: folding it in makes a refit mint a
    new lineage on every run, and reconstruction then never converges."""
    return hidden_lineage_reads(code)


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


#: Spellings a statement must contain to read the environment at all: the
#: cheap test before parsing it (`statement_environment_reads`).
_ENVIRONMENT_MARKERS = ("environ", "getenv", "getcwd")


def statement_environment_reads(code: str, user_ns: Mapping[str, Any] | None = None) -> set[tuple[str, str]]:
    """The environment reads written in *code* whose value a key can fold
    (`cash.effects.environment_input`): ``os.getenv("NAME")``,
    ``os.environ["NAME"]``, ``os.getcwd()``.

    Only the statement's own text: a read inside a function it calls is not
    seen here.
    """
    if not code or not any(marker in code for marker in _ENVIRONMENT_MARKERS):
        return set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    found = set()
    for node in ast.walk(tree):
        entry = environment_input(node, user_ns)
        if entry is not None:
            found.add(entry)
    return found


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
    return environment_component(statement_environment_reads(code, user_ns))


def callable_source_component(function_tracker: Any, inputs: set[str], user_ns: dict) -> str:
    """``:name:hash`` for every user-defined callable among *inputs*."""
    if function_tracker is None:
        return ""
    hashes = function_tracker.get_callable_source_hashes(inputs, user_ns)
    if not hashes:
        return ""
    return ":" + ":".join(f"{k}:{v}" for k, v in sorted(hashes.items()))


def _tracked(function_tracker: Any) -> set[str]:
    return getattr(function_tracker, "tracked_modules", None) or set()


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
    parents = getattr(function_tracker, "dep_file_to_parents", None) or {}
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
    `load` (round 27, r27s2 -- the same fix as `module_read_lineage`, for the
    other common spelling). Without one, or when the closure cannot be
    bounded, the whole module, as before.
    """
    if module_name not in _tracked(function_tracker):
        return ""
    mod_obj = sys.modules.get(module_name)
    mod_file = getattr(mod_obj, "__file__", None) if mod_obj else None
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
    if isinstance(value, types.ModuleType):
        mod_file = getattr(value, "__file__", None)
        # By the module's OWN name, not the name the cell bound it to.
        # `tracked_modules` holds real module names, so `import tickets_lib
        # as tl` failed this test and returned "" -- the module's source
        # stayed out of `tl`'s lineage, `tl` keyed identically before and
        # after an edit to tickets_lib.py, and every statement built on it
        # kept its cached value. Reloading worked, the badge said MODULE
        # RELOADED, and the cell still returned the pre-edit answer until the
        # kernel was restarted (round 27, r27s2, 3/3, with two exported
        # deliverables computed from a number the user had just fixed).
        #
        # `var_name` is still accepted so a tracker that registered the bound
        # name keeps working; for an unaliased import the two are equal, so
        # no existing lineage moves.
        names = {getattr(value, "__name__", var_name), var_name}
        if not (mod_file and os.path.isfile(mod_file) and names & _tracked(function_tracker)):
            return ""
        parents = getattr(function_tracker, "dep_file_to_parents", None) or {}
        dep_files = {dep for dep, owners in parents.items() if names & set(owners)}
        digest = read_module_source_hash(mod_file, dep_files)
        return f":mod_src:{digest}" if digest else ""

    if callable(value):
        obj_module = getattr(value, "__module__", None)
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

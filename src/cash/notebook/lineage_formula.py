"""The one formula for a variable's lineage, shared by both engines.

The runtime (``statement/lineage.py``) records a variable's lineage when its
statement runs; the upstream simulator (``upstream/virtual_lineage.py``)
recomputes it from the code above a cell. Wherever the two disagree, a cell
reading the variable sees "changed" with nothing changed and re-runs other
cells' statements to repair it. They used to build the lineage separately, and
in round 21 they had drifted apart: a name imported from a local module carried
that module's source hash at runtime and not in the simulation, so every cell
downstream of ``from helpers import clean`` re-ran statements even in a plain
top-to-bottom run -- among them a model refit, which made an "accuracy before"
report use the refitted model.

Everything here is a pure function of its arguments. The callers decide which
ingredients they have; this module decides how they combine.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import sys
import types
from typing import Any, Callable, Iterable

from ..tracking.module_symbols import closure_digest, static_attribute_reads

logger = logging.getLogger(__name__)


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
) -> str:
    """The lineage hash of one output of a statement."""
    lineage_str = f"{source_hash}:{':'.join(sorted(input_lineages))}{file_component}{func_component}{module_component}"
    return hashlib.sha256(lineage_str.encode("utf-8")).hexdigest()


def callable_source_component(function_tracker: Any, inputs: set[str], user_ns: dict) -> str:
    """``:name:hash`` for every user-defined callable among *inputs*."""
    if function_tracker is None:
        return ""
    hashes = function_tracker.get_callable_source_hashes(inputs, user_ns)
    if not hashes:
        return ""
    return ":" + ":".join(f"{k}:{v}" for k, v in sorted(hashes.items()))


def _tracked(function_tracker: Any) -> set[str]:
    return getattr(function_tracker, "_tracked_modules", None) or set()


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
    parents = getattr(function_tracker, "_dep_file_to_parents", None) or {}
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
        # `_tracked_modules` holds real module names, so `import tickets_lib
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
        parents = getattr(function_tracker, "_dep_file_to_parents", None) or {}
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

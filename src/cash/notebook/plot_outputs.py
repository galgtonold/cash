"""Showing what a plot drew last time, when nothing it reads has changed.

A statement that draws through pyplot runs every time (``pyplot_state``):
the figure it leaves is process-global state no key holds. Under the inline
backend, though, a plot usually ends where it began: ``plt.show()``, or the
end of the cell, renders the open figures into the cell's output and closes
them. From "no figure open" to "no figure open", the statements in between
(a *plot run*) leave behind only what they showed, and the names they bound.

So a plot run is kept as what it showed: per statement, its printed output
and its displays (the rendered image among them), and what the cell's last
line echoed, as the value. Run again with the same code and inputs, under
the same matplotlib settings, it is served: the output is shown again and
the statements are skipped. The figure is then not live, so a plot run that
binds a name (``fig, ax = plt.subplots()``) is never kept: served, it would
leave the name unbound where plain Python binds it. Nor is one whose last
line echoed a figure, axes or artist, or one that wrote or read a file,
drew random numbers, changed matplotlib's settings or anything else a
skipped run would not do: it runs every time, as before.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from cash._user_attrs import attribute_or

from .pyplot_state import CHANGED_PYPLOT_REASON

logger = logging.getLogger(__name__)

__all__ = [
    "PLOT_FACTS",
    "PLOT_RECORD_CAP",
    "PlotFacts",
    "figure_settings_digest",
    "inline_backend_closes_figures",
    "is_pyplot_reason",
    "plot_shapes_key",
    "rich_records",
    "statement_plot_facts",
    "first_read_after",
]

#: The metric field a statement's :class:`PlotFacts` travel in, from the
#: statement processor to the cell executor, which takes it out.
PLOT_FACTS = "plot_facts"

#: The most output, in bytes, a kept plot run may have shown. Over it, it runs.
PLOT_RECORD_CAP = 16 * 1024 * 1024

#: How the inline backend is named: by its module, or (matplotlib 3.9+) by its short name.
_INLINE_BACKENDS = frozenset({"module://matplotlib_inline.backend_inline", "inline"})


@dataclass(frozen=True)
class PlotFacts:
    """What one statement did that decides whether a plot run holding it can
    be skipped: why it was not stored, what a skipped run would not do
    (*blockers*), and the names it bound."""

    reasons: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    bound: frozenset[str] = field(default_factory=frozenset)
    #: The environment variables library code read as it ran, with what they
    #: held (`environment_digests`): a plot run is served only while they
    #: hold the same.
    env: Mapping[str, str] = field(default_factory=dict)
    #: Where it left the global random streams it moved or seeded: a served
    #: plot run puts them there, as a hit does.
    rng_state: Mapping[str, Any] = field(default_factory=dict)
    #: The streams it seeded as it ran.
    rng_reseeded: frozenset[str] = frozenset()
    #: The streams it drew from without seeding them first: what it drew
    #: depends on where they stood, so a plot run holding it is served only
    #: while they stand at the same lineage.
    rng_read: frozenset[str] = frozenset()
    #: The streams it seeded with a value before drawing from them.
    rng_seeded_first: frozenset[str] = frozenset()
    #: ``(records, changes)`` of the helper memos it called
    #: (`helper_memos.used`): a served plot run must find what it read
    #: there and puts back what it added, as a hit does.
    memos: tuple[Any, Any] | None = None


def is_pyplot_reason(reason: object) -> bool:
    """Whether *reason*, why a statement was not stored, is only that it
    draws through pyplot: a ``plt.*`` call, a helper that does, a change to
    pyplot's figures, or an output holding a figure or axes."""
    text = str(reason)
    if text == CHANGED_PYPLOT_REASON:
        return True
    if text.startswith("Side effect: ") and text.endswith("(display)"):
        return True
    return "pyplot's current figure" in text


#: The global streams whose position a lineage follows: a plot run that
#: draws from another one without seeding it first is not kept.
_LINEAGED_STREAMS = frozenset({"random", "numpy.random"})


def statement_plot_facts(
    run: Any,
    execution: Any,
    *,
    drew: Iterable[str],
    seeded_first: Iterable[str],
    unplaced: bool,
) -> PlotFacts:
    """The :class:`PlotFacts` of the statement *run*, which ran as *execution*.

    *drew*: the global streams it moved; *seeded_first*: those it seeded
    with a value before drawing from them; *unplaced*: it drew from a stream
    the upstream check could not put where a run from the top has it."""
    blockers: list[str] = []
    # A write from a line the user waived (``# @cash:assume-safe``) is one a
    # skipped run may leave out.
    written = set(execution.written_paths or ()) - set(execution.waived_paths or ())
    if written or execution.unseen_writer or execution.children_running:
        blockers.append("writes files or starts processes")
    files = sorted(
        {path for path in execution.accessed_files or () if not _drawing_file(path)}
        | set(execution.accessed_remote or ())
    )
    if files:
        # What it read is in no key of a plot run.
        blockers.append(f"reads files ({', '.join(files[:3])})")
    seeded_first = frozenset(seeded_first)
    reseeded = frozenset(execution.rng_reseeded or ())
    moved = dict(execution.rng_moved or {})
    streams_read = frozenset((set(moved) | set(drew)) - seeded_first)
    if reseeded - seeded_first:
        # Seeded after a draw, or from the system's entropy: not a position
        # its code and inputs alone decide.
        blockers.append("seeds a random stream other than before its draws")
    elif streams_read - _LINEAGED_STREAMS:
        blockers.append(f"draws random numbers ({', '.join(sorted(streams_read - _LINEAGED_STREAMS))})")
    elif streams_read and unplaced:
        blockers.append("draws from a random stream cash could not place")
    memos = attribute_or(run, "plot_memos", None)
    if memos is None and attribute_or(run, "helper_memos", None):
        # Stored on its own: its entry carries them, as the plot run must.
        memos = (run.helper_memos, attribute_or(run, "memo_changes", None))
    if isinstance(memos, str):
        blockers.append(memos)
    if run.state_modules or run.process_state or run.rebound_modules:
        blockers.append("changes module state")
    if attribute_or(run, "callee_globals", None):
        blockers.append("sets globals through a function")
    reasons = tuple(str(r) for r in run.metrics.get("uncacheable_reasons") or ())
    return PlotFacts(
        reasons,
        tuple(blockers),
        frozenset(run.outputs or ()),
        dict(run.env_reads or {}),
        rng_state=moved,
        rng_reseeded=reseeded,
        rng_read=streams_read,
        rng_seeded_first=seeded_first,
        memos=memos if isinstance(memos, tuple) else None,
    )


#: Files a renderer reads to draw text: the fonts matplotlib loads the first
#: time it draws with one.
_FONT_SUFFIXES = (".ttf", ".otf", ".ttc", ".afm", ".pfb", ".pfa", ".woff", ".woff2")


def _drawing_file(path: str) -> bool:
    """Whether *path* is one matplotlib reads to draw, not the plot's data:
    a font, or a file of its own (its style sheets, its font cache)."""
    if str(path).lower().endswith(_FONT_SUFFIXES):
        return True
    matplotlib = sys.modules.get("matplotlib")
    if matplotlib is None:
        return False
    try:
        own = (matplotlib.get_data_path(), matplotlib.get_cachedir())
    except Exception:  # noqa: BLE001 - not known to be its own
        return False
    return any(folder and str(path).startswith(str(folder)) for folder in own)


#: How cash says a statement changed a value in place (``ax.plot(...)``).
_IN_PLACE = re.compile(r"^In-place mutation on: ([\w, ]+?) \(")


def mutated_names(reason: object) -> frozenset[str] | None:
    """The names *reason* says the statement changed in place, or None when
    it says something else."""
    found = _IN_PLACE.match(str(reason))
    if found is None:
        return None
    return frozenset(name.strip() for name in found.group(1).split(",") if name.strip())


def inline_backend_closes_figures() -> bool:
    """Whether pyplot draws through the inline backend, which renders the
    open figures into the cell's output and closes them after it shows
    them. Nothing else is taken to end a plot."""
    matplotlib = sys.modules.get("matplotlib")
    if matplotlib is None or "matplotlib.pyplot" not in sys.modules:
        return False
    try:
        # Read past ``RcParams.__getitem__``, which resolves a backend not yet chosen.
        backend = dict.get(matplotlib.rcParams, "backend")
    except Exception:  # noqa: BLE001 - settings that cannot be read end nothing
        return False
    if backend not in _INLINE_BACKENDS:
        return False
    inline = _inline_settings()
    return inline is not None and bool(attribute_or(inline, "close_figures", False))


def _inline_settings() -> Any:
    """The inline backend's settings object, when it is loaded."""
    module = sys.modules.get("matplotlib_inline.config")
    config = attribute_or(module, "InlineBackend", None)
    if config is None:
        return None
    try:
        return config.instance() if config.initialized() else None
    except Exception:  # noqa: BLE001 - a half-set-up backend: no inline config
        return None


def figure_settings_digest(shell: Any) -> str | None:
    """What decides how the same drawing renders: matplotlib's version and
    settings (``rcParams``, a style, a seaborn theme), the inline backend's
    formats, and which formats the notebook shows. None when they cannot be
    read: nothing is kept or served then."""
    matplotlib = sys.modules.get("matplotlib")
    if matplotlib is None:
        return None
    try:
        params = sorted((key, repr(value)) for key, value in dict.items(matplotlib.rcParams))
        inline = _inline_settings()
        inline_part = (
            sorted((name, repr(value)) for name, value in inline.trait_values(config=True).items())
            if inline is not None
            else None
        )
        formatter = attribute_or(shell, "display_formatter", None)
        formats = sorted(attribute_or(formatter, "active_types", None) or ())
        text = repr((matplotlib.__version__, params, inline_part, formats))
    except Exception:
        logger.debug("could not read matplotlib's settings", exc_info=True)
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def plot_shapes_key(first_code: str, occurrence: int) -> str:
    """Where the plot runs starting with *first_code* (its *occurrence* in
    the cell) are listed, as their statements' texts."""
    digest = hashlib.sha256(f"{occurrence}:{first_code}".encode()).hexdigest()
    return f"plotshape:{digest}"


def rich_records(outputs: Iterable[Any]) -> list[dict] | None:
    """*outputs*, a statement's displays, as plain display data to publish
    again; None when one is not plain data (a widget, an object)."""
    records: list[dict] = []
    for output in outputs or ():
        if isinstance(output, dict):
            data, metadata = output.get("data"), output.get("metadata")
        else:
            data, metadata = attribute_or(output, "data", None), attribute_or(output, "metadata", None)
        if not isinstance(data, dict) or not all(isinstance(v, (str, dict, list)) for v in data.values()):
            return None
        records.append({"data": dict(data), "metadata": dict(metadata or {})})
    return records


def record_size(records: Iterable[dict]) -> int:
    """About how many bytes *records* hold."""
    return sum(len(str(v)) for record in records for v in record.get("data", {}).values())


# ---------------------------------------------------------------------------
# What the code after a plot run reads
# ---------------------------------------------------------------------------

#: Calls that reach a name by a string: a statement making one is searched
#: as text for the names a plot run bound.
_BY_STRING = frozenset({"eval", "exec", "globals", "locals", "vars", "get_ipython", "getattr", "__import__"})

#: IPython's output history: a later read of it may read an echoed value.
_HISTORY = frozenset({"_", "__", "___", "Out", "_oh"})
_HISTORY_ENTRY = re.compile(r"^_\d+$")
_HISTORY_TEXT = r"\bOut\b|\b_oh\b|\b_\d+\b|(?<![\w.])_{1,3}(?!\w)"


def first_read_after(names: frozenset[str], statements: Iterable[tuple[str, ast.stmt | None]]) -> str | None:
    """The first of *names* the *statements* (``(source, tree)`` in run
    order; tree None when it cannot be parsed) read before binding it again,
    or None when none does.

    A statement that cannot be parsed, or reaches names by a string, is read
    as text: any mention counts. Binding inside a branch or a loop does not
    count as binding again.
    """
    watched = set(names)
    for source, node in statements:
        if not watched:
            return None
        if node is None or _reaches_by_string(node):
            for name in sorted(watched):
                pattern = _HISTORY_TEXT if name in _HISTORY else rf"\b{re.escape(name)}\b"
                if re.search(pattern, source):
                    return name
            continue
        read = _free_reads(node)
        for name in read:
            if name in watched or (_HISTORY_ENTRY.match(name) and watched & _HISTORY):
                return name
        watched -= _bound_for_good(node)
    return None


def _reaches_by_string(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            if name in _BY_STRING:
                return True
    return False


def _bound_for_good(node: ast.stmt) -> set[str]:
    """The names *node*, a top-level statement, always binds."""
    bound: set[str] = set()
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            bound |= _target_names(target)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        bound.add(node.name)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            bound.add((alias.asname or alias.name).split(".")[0])
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            if item.optional_vars is not None:
                bound |= _target_names(item.optional_vars)
    return bound


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names |= _target_names(element)
        return names
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return set()


def _free_reads(node: ast.AST) -> set[str]:
    """The names *node* reads (or deletes) from the notebook's namespace,
    before it binds them itself where that is certain: an assignment's
    value is read before its targets are bound, a ``for`` loop's and a
    ``with`` statement's body reads the names they bound. A function's or a
    comprehension's own names are its own."""
    reader = _Reads()
    reader.visit(node)
    return reader.reads


class _Reads(ast.NodeVisitor):
    def __init__(self) -> None:
        self.reads: set[str] = set()
        self._local: list[set[str]] = [set()]

    def _read(self, name: str) -> None:
        if not any(name in scope for scope in self._local):
            self.reads.add(name)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Load, ast.Del)):
            self._read(node.id)
        else:
            self._local[-1].add(node.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._read(node.target.id)
        self.visit(node.target)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        # The loop may not run: what it binds is bound only inside it.
        self.visit(node.iter)
        self._scoped(set(), [node.target, *node.body])
        self._scoped(set(), node.orelse)

    visit_AsyncFor = visit_For

    def visit_If(self, node: ast.If | ast.While) -> None:
        # What a branch binds is bound only in that branch.
        self.visit(node.test)
        self._scoped(set(), node.body)
        self._scoped(set(), node.orelse)

    visit_While = visit_If

    def visit_Try(self, node: Any) -> None:
        self._scoped(set(), node.body)
        for handler in node.handlers:
            self._scoped(set(), [handler])
        self._scoped(set(), node.orelse)
        self._scoped(set(), node.finalbody)

    visit_TryStar = visit_Try

    def _scoped(self, local: set[str], nodes: Iterable[ast.AST]) -> None:
        self._local.append(set(local))
        try:
            for sub in nodes:
                self.visit(sub)
        finally:
            self._local.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> None:
        args = node.args
        for default in [*args.defaults, *(d for d in args.kw_defaults if d is not None)]:
            self.visit(default)
        if not isinstance(node, ast.Lambda):
            for decorator in node.decorator_list:
                self.visit(decorator)
            self._local[-1].add(node.name)
        params = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
        params |= {a.arg for a in (args.vararg, args.kwarg) if a is not None}
        body = node.body if isinstance(node.body, list) else [node.body]
        # Names the body assigns are its own, unless declared global.
        stored = {
            sub.id
            for stmt in body
            for sub in ast.walk(stmt)
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store)
        }
        declared = {
            name for stmt in body for sub in ast.walk(stmt) if isinstance(sub, ast.Global) for name in sub.names
        }
        self._scoped(params | (stored - declared), body)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def _comprehension(self, node: Any, elements: list[ast.AST]) -> None:
        first, *rest = node.generators
        self.visit(first.iter)
        local: set[str] = set()
        for generator in node.generators:
            local |= _target_names(generator.target)
        parts: list[ast.AST] = [*first.ifs]
        for generator in rest:
            parts.extend([generator.iter, *generator.ifs])
        self._scoped(local, [*parts, *elements])

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._comprehension(node, [node.elt])

    visit_SetComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._comprehension(node, [node.key, node.value])

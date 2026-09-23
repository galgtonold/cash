"""What a statement does outside its variables: files, the network, the console.

Pure AST. :class:`_SideEffectVisitor` flags the calls ``NOTEBOOK_POLICY``
refuses to cache; :func:`statement_write_repeatability` tells a write that
replaces its file from one that appends to it. Resolving the paths a statement
reads or writes needs the live namespace and lives in
:mod:`cash.analysis.namespace_effects`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..effects import Action, EffectKind, classify_call, is_open_write_mode, writes_to_console

__all__ = [
    "NOTEBOOK_POLICY",
    "SCANNED_KINDS",
    "SideEffectInfo",
    "REPEATABILITY_REPLACING",
    "REPEATABILITY_ACCUMULATING",
    "REPEATABILITY_UNKNOWN",
    "statement_write_repeatability",
    "READ_TEXT_MARKERS",
    "get_call_name",
    "get_call_module",
    "get_base_name",
]


# ---------------------------------------------------------------------------
# Side-effect detection
# ---------------------------------------------------------------------------

#: What a notebook statement does about each kind of effect in it. What a kind
#: IS lives in :mod:`cash.effects`, shared with the decorator, whose own table
#: is ``cash.purity_analyzer.DECORATOR_POLICY``. The two differ where the paths
#: differ on purpose -- a notebook replays a statement's printed output, so a
#: print caches here, while a decorator hit drops it -- and a test keeps both
#: covering every kind, so a new kind cannot slip past either.
NOTEBOOK_POLICY: dict[EffectKind, Action] = {
    EffectKind.FILE_WRITE: Action.REFUSE,
    EffectKind.FILE_READ: Action.CACHE_AS_INPUT,
    # A slow GET is a result worth keeping, like reading a file.
    EffectKind.NETWORK_READ: Action.CACHE,
    EffectKind.NETWORK_WRITE: Action.REFUSE,
    EffectKind.NETWORK: Action.REFUSE,
    # A query's answer is kept the same way; `# @cash:ttl=N` bounds its age.
    EffectKind.DB_READ: Action.CACHE,
    EffectKind.DB_WRITE: Action.REFUSE,
    EffectKind.SUBPROCESS: Action.REFUSE,
    EffectKind.CLOCK: Action.REFUSE,
    # The value of a read whose name is written out goes into the key
    # (`compute_cache_key`), so a new value is a new entry.
    EffectKind.ENVIRONMENT: Action.CACHE_AS_INPUT,
    # The captured output is replayed on a hit.
    EffectKind.CONSOLE: Action.CACHE,
    EffectKind.DISPLAY: Action.REFUSE,
    EffectKind.INTERACTIVE: Action.REFUSE,
}


#: Kinds judged by :meth:`CodeAnalyzer.scan_for_forbidden_functions`, which
#: resolves a call through what its names are bound to (``from time import time
#: as now; now()``). The side-effect visitor below reads spelling alone.
SCANNED_KINDS: frozenset[EffectKind] = frozenset({EffectKind.CLOCK, EffectKind.ENVIRONMENT, EffectKind.INTERACTIVE})


#: How a refused kind is named in its skip reason, ``Side effect: <call> (<label>)``.
_NOTEBOOK_LABELS: dict[EffectKind, str] = {
    EffectKind.FILE_WRITE: "file_write",
    EffectKind.NETWORK_WRITE: "network",
    EffectKind.NETWORK: "network",
    EffectKind.DB_WRITE: "database_write",
    EffectKind.SUBPROCESS: "system",
    EffectKind.DISPLAY: "display",
}


@dataclass
class SideEffectInfo:
    """Information about a detected side effect."""

    kind: str  # the label shown in a skip reason: 'file_write', 'network', 'system', ...
    description: str
    line: int = 0
    effect_kind: EffectKind | None = None


# Cheap textual pre-filter for statement_writes_files: superset of the names
# in the write-detection tables above, checked before any AST work.
_WRITE_TEXT_MARKERS: tuple[str, ...] = (
    "open(",
    "write",
    "to_",
    "save",
    "dump",
    "os.",
    "shutil.",
    "mkdir",
    "touch",
    "link",
)


def _is_append_mode_call(call: ast.Call) -> bool:
    """True when *call* carries a statically-visible APPEND mode string.

    Covers ``open(p, 'a')`` (mode is positional arg 1) and any writer taking a
    ``mode=`` keyword (``open(p, mode='a')``, ``df.to_csv(p, mode='a')``). A
    non-literal mode (``open(p, m)``) is NOT provable and returns False.
    """
    if isinstance(call.func, ast.Name) and call.func.id == "open" and len(call.args) >= 2:
        mode_arg = call.args[1]
        if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
            if "a" in mode_arg.value:
                return True
    for kw in call.keywords:
        if (
            kw.arg == "mode"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
            and "a" in kw.value.value
        ):
            return True
    return False


# Write calls that REPLACE their target wholesale, so re-running one lands the
# same bytes. ``to_hdf`` is deliberately ABSENT: pandas defaults it to
# ``mode='a'``, making it accumulating despite its truncating siblings.
_REPLACING_WRITE_METHODS: frozenset[str] = frozenset(
    {
        "to_csv",
        "to_excel",
        "to_parquet",
        "to_json",
        "to_pickle",
        "to_feather",
        "to_stata",
        "to_latex",
        "to_html",
        "to_markdown",
        "savefig",  # matplotlib truncates the PNG
        "save",  # numpy / PIL / torch all truncate
        "write_text",  # pathlib truncates
        "write_bytes",
    }
)


# Module-level writers that land the same result when repeated. Every other
# module-level file write (remove/rename/move/mkdir/rmtree...) is NOT
# repeatable -- a second run raises or acts on a target that is already gone.
_REPLACING_IO_FUNCTIONS: frozenset[str] = frozenset(
    {
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
    }
)


REPEATABILITY_REPLACING = "replacing"


REPEATABILITY_ACCUMULATING = "accumulating"


REPEATABILITY_UNKNOWN = "unknown"


# Module writers that take an already-open FILE HANDLE rather than a path, so
# their repeatability is decided by whatever opened it -- never by the call
# itself. Maps the writer's name -> positional index of the handle argument.
_HANDLE_WRITE_FUNCTIONS: dict[str, int] = {
    "json.dump": 1,
    "pickle.dump": 1,
}


def _defers_to_open(node: ast.expr | None, local_handles: frozenset[str]) -> bool:
    """True when *node* is a handle whose ``open()`` is in this same statement.

    The ``open()`` carries the mode and is walked separately, so the write on
    the handle must contribute NO verdict of its own -- otherwise its UNKNOWN
    outranks the ``open()``'s provable one and a plain truncating write
    (``with open(p, 'wb') as f: pickle.dump(obj, f)``) is misread as unsafe to
    repeat.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
        return True
    return isinstance(node, ast.Name) and node.id in local_handles


def _open_mode_node(call: ast.Call) -> ast.expr | None:
    """The mode argument of an ``open()`` call, positional or keyword."""
    if len(call.args) >= 2:
        return call.args[1]
    for kw in call.keywords:
        if kw.arg == "mode":
            return kw.value
    return None


def _locally_opened_handles(tree: ast.AST) -> set[str]:
    """Names bound to an ``open()`` handle WITHIN this statement.

    ``with open(p, 'w') as f: f.write(x)`` and ``f = open(p, 'w'); f.write(x)``
    both carry their mode on the ``open()`` call, which is walked separately.
    Without this, the ``f.write`` would contribute an UNKNOWN that outranks the
    ``open()``'s provable verdict and mislabels a plain truncating write.
    """
    handles: set[str] = set()

    def _is_open(node) -> bool:
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open"

    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if _is_open(item.context_expr) and isinstance(item.optional_vars, ast.Name):
                    handles.add(item.optional_vars.id)
        elif isinstance(node, ast.Assign) and _is_open(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    handles.add(target.id)
    return handles


def _call_repeatability(call: ast.Call, local_handles: frozenset[str] = frozenset()) -> str | None:
    """Repeatability of one call node, or ``None`` if it is not a file write."""
    func = call.func
    if writes_to_console(call):
        return None
    if isinstance(func, ast.Name) and func.id == "open":
        mode = _open_mode_node(call)
        if mode is None:
            return None  # no mode argument -> defaults to 'r', a read
        if not (isinstance(mode, ast.Constant) and isinstance(mode.value, str)):
            # A computed mode proves nothing. `is_open_write_mode` reports
            # False here, which the write-DETECTION path reads as "not a
            # write" -- but a `.write` on this handle still makes the statement
            # a writer, and the mode could be 'a' at runtime.
            return REPEATABILITY_UNKNOWN
        if not is_open_write_mode(call):
            return None  # provably a read mode
        return REPEATABILITY_ACCUMULATING if "a" in mode.value else REPEATABILITY_REPLACING
    if isinstance(func, ast.Attribute):
        method = func.attr
        effect = classify_call(call)
        if effect is None or not effect.method or effect.kind is not EffectKind.FILE_WRITE:
            return None
        if _is_append_mode_call(call):
            return REPEATABILITY_ACCUMULATING
        if method in _REPLACING_WRITE_METHODS:
            return REPEATABILITY_REPLACING
        # ``f.write(...)`` / ``f.writelines(...)``: the mode lives on whatever
        # opened the handle, never on the write itself. If that ``open()`` is in
        # this same statement, defer -- its own node is walked separately and
        # carries the provable verdict. A handle bound in an earlier cell is
        # genuinely unresolvable from here.
        if _defers_to_open(func.value, local_handles):
            return None
        return REPEATABILITY_UNKNOWN
    if isinstance(func, ast.Name):
        return None
    return None


def statement_write_repeatability(code: str, tree: "ast.Module | None" = None) -> str:
    """How safe is it to re-run *code*'s file writes?

    The question the write-detection helpers above do not answer:
    :func:`cash.effects.is_open_write_mode` pools ``'a'`` with ``'w'``, so every consumer learns
    only "this writes a file", never "repeating this write duplicates data".

    Returns the WORST verdict across every write in the statement:

    * ``'accumulating'`` -- provably appends; re-firing duplicates the payload
    * ``'replacing'``    -- provably truncates; re-firing lands the same bytes
    * ``'unknown'``      -- cannot tell (variable mode, a handle opened
      elsewhere, ``os.rename``/``shutil.move``, ...)

    A statement with no recognised write is ``'replacing'``: there is nothing to
    repeat, so it never constrains the planner.
    """
    if tree is None:
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, TypeError):
            return REPEATABILITY_UNKNOWN
    local_handles = frozenset(_locally_opened_handles(tree))
    verdicts: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        verdict = _call_repeatability(node, local_handles)
        if verdict is not None:
            verdicts.add(verdict)
        # Module-level writers (os.remove, shutil.move, ...) are recognised
        # by their full name rather than the call shapes above.
        effect = classify_call(node)
        if effect is None or effect.method or effect.kind is not EffectKind.FILE_WRITE or "." not in effect.name:
            continue
        key = effect.name
        if key in _REPLACING_IO_FUNCTIONS:
            continue
        handle_idx = _HANDLE_WRITE_FUNCTIONS.get(key)
        if handle_idx is not None:
            handle = node.args[handle_idx] if len(node.args) > handle_idx else None
            if _defers_to_open(handle, local_handles):
                continue  # the open() decides; it is walked separately
        verdicts.add(REPEATABILITY_UNKNOWN)
    if REPEATABILITY_ACCUMULATING in verdicts:
        return REPEATABILITY_ACCUMULATING
    if REPEATABILITY_UNKNOWN in verdicts:
        return REPEATABILITY_UNKNOWN
    return REPEATABILITY_REPLACING


# Cheap textual pre-filter for statement_read_paths.
READ_TEXT_MARKERS: tuple[str, ...] = ("open(", "read", "load")


def get_call_name(func_node: ast.AST) -> str | None:
    """Extract the function name from a call's func node."""
    if isinstance(func_node, ast.Name):
        return func_node.id
    if isinstance(func_node, ast.Attribute):
        return func_node.attr
    return None


def get_call_module(func_node: ast.AST) -> str | None:
    """Extract the module/object prefix from a call's func node."""
    if isinstance(func_node, ast.Attribute):
        if isinstance(func_node.value, ast.Name):
            return func_node.value.id
        if isinstance(func_node.value, ast.Attribute):
            # e.g., os.path.join -> module = 'os.path'
            parts: list[str] = []
            node: ast.AST = func_node.value
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
            return ".".join(reversed(parts))
    return None


def get_base_name(node: ast.AST) -> str | None:
    """Extract a human-readable name for the object a method is called on."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = get_base_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Subscript):
        base = get_base_name(node.value)
        return f"{base}[...]" if base else None
    return None


class _SideEffectVisitor(ast.NodeVisitor):
    """Collects the calls a notebook statement must not be restored past:
    those whose kind :data:`NOTEBOOK_POLICY` refuses."""

    def __init__(self) -> None:
        self.effects: list[SideEffectInfo] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Detect function/method calls with side effects."""
        effect = classify_call(node)
        if effect is not None and effect.kind not in SCANNED_KINDS and NOTEBOOK_POLICY[effect.kind] is Action.REFUSE:
            if effect.method:
                base = get_base_name(node.func.value)  # type: ignore[attr-defined]
                description = f"{base + '.' if base else ''}{effect.name}()"
            elif effect.name == "open":
                description = "open() with write mode"
            else:
                description = f"{effect.name}()"
            self.effects.append(
                SideEffectInfo(
                    kind=_NOTEBOOK_LABELS.get(effect.kind, effect.kind.value),
                    description=description,
                    line=getattr(node, "lineno", 0),
                    effect_kind=effect.kind,
                )
            )
        self.generic_visit(node)

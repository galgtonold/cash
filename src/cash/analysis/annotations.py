"""Parser for @cash: comment annotations that control per-statement caching."""

from __future__ import annotations

import ast
import re
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..diagnostics import warn_diagnostic
from ..exceptions import CashCacheIneffectiveWarning

__all__ = [
    "CacheAnnotation",
    "ANNOTATION_PATTERN",
    "KNOWN_DIRECTIVES",
    "leading_cell_annotation",
    "parse_annotation_line",
    "parse_annotations_in_range",
    "get_statement_annotations",
    "extract_annotations_for_statements",
    "ASSUME_SAFE_RE",
    "audited_lines",
    "assume_safe_block_lines",
    "is_assume_safe_block",
    "waiver_items",
]


@dataclass
class CacheAnnotation:
    """Represents cache control annotations for a statement."""

    persist: bool = False  # Force disk persistence
    no_cache: bool = False  # Disable caching entirely
    ttl: int | None = None  # Override TTL in seconds
    allow_random: bool = False  # Suppress randomness warnings
    cache_fit: bool = False  # Opt in to caching a bare ``estimator.fit(X, y)``
    no_cache_calls: bool = False  # Opt OUT of caching CALLS inside the statement
    # Cache the statement despite its side effects: a hit skips them. For a
    # call that only looks like a write -- a POST that runs a query. Never
    # spread from a cell's header (see ``leading_cell_annotation``).
    assume_safe: bool = False

    def merge(self, other: CacheAnnotation) -> CacheAnnotation:
        """Merge with another annotation (other takes precedence for ttl)."""
        return CacheAnnotation(
            persist=self.persist or other.persist,
            no_cache=self.no_cache or other.no_cache,
            ttl=other.ttl if other.ttl is not None else self.ttl,
            allow_random=self.allow_random or other.allow_random,
            cache_fit=self.cache_fit or other.cache_fit,
            no_cache_calls=self.no_cache_calls or other.no_cache_calls,
            assume_safe=self.assume_safe or other.assume_safe,
        )

    def has_directives(self) -> bool:
        """Check if any directives are set."""
        return (
            self.persist
            or self.no_cache
            or self.ttl is not None
            or self.allow_random
            or self.cache_fit
            or self.no_cache_calls
            or self.assume_safe
        )


# Regex patterns for annotation parsing ([\w-]+ allows hyphens in directive names)
# Whitespace is tolerated after the colon and around ``=`` so both the
# documented ``# @cash:no-cache`` and the natural ``# @cash: no-cache`` parse
# (see test_annotation_with_spaces). Without the ``\s*`` after the colon the
# spaced form was silently ignored -- the statement was cached as normal.
#
# The value group is ``\S*``, NOT ``\d+``. An unanchored ``\d+`` matched the
# leading digit run and silently dropped the rest, so ``ttl=5m`` parsed as
# ``ttl=5`` -- five SECONDS where five minutes was asked for, a 60x error, and
# ``ttl=1h`` a 3600x one. The only symptom is a cache that keeps missing, which
# reads as "cash isn't working" rather than "my annotation was truncated".
# Capturing the whole token lets the directive handler see ``5m``
# and reject it out loud.
ANNOTATION_PATTERN = re.compile(r"#\s*@cash:\s*([\w-]+)(?:\s*=\s*(\S*))?")

#: Every directive name Cash acts on. ``assume-safe`` is also read by the
#: purity analyser, from a decorated function's source.
KNOWN_DIRECTIVES: tuple[str, ...] = (
    "no-cache",
    "persist",
    "ttl",
    "allow-random",
    "cache-fit",
    "no-cache-calls",
    "assume-safe",
)

#: Unknown directive names already warned about in this process, so a typo in a
#: cell that is parsed on every run (and by the upstream checker for every cell
#: below it) is reported once, not every time.
_warned_unknown_directives: set[str] = set()


def _squash(name: str) -> str:
    return name.replace("-", "").replace("_", "")


def _warn_unknown_directive(directive: str, lineno: int | None) -> None:
    """Say, once per name per session, that ``# @cash:<directive>`` does nothing.

    An unknown directive used to drop silently, which is how an old
    ``# @cash:nocache`` (the run-together spelling, since removed) went on
    caching the statement it was written to stop.

    With *lineno* (the line in the cell) the warning is blamed on ``<cash>``
    at that line, like the other notebook warnings: the nearest frame outside
    Cash while a cell runs is ipykernel's, which tells the reader nothing.
    """
    if directive in _warned_unknown_directives:
        return
    _warned_unknown_directives.add(directive)
    suggestion = next((known for known in KNOWN_DIRECTIVES if _squash(known) == _squash(directive)), None)
    if suggestion is not None:
        what = (
            f"`# @cash:{directive}` is not a directive Cash knows (did you mean "
            f"`# @cash:{suggestion}`?), so it was IGNORED and the statement is "
            f"cached as if the comment were not there."
        )
        fix = f"write it as `# @cash:{suggestion}`."
    else:
        what = (
            f"`# @cash:{directive}` is not a directive Cash knows, so it was "
            f"IGNORED and the statement is cached as if the comment were not there."
        )
        fix = "use one of: " + ", ".join(f"`{d}`" for d in KNOWN_DIRECTIVES) + "."
    location = None if lineno is None else ("<cash>", lineno)
    warn_diagnostic(CashCacheIneffectiveWarning, "ANNOT-UNKNOWN-DIRECTIVE", what, fix, location=location)


def parse_annotation_line(line: str, lineno: int | None = None) -> CacheAnnotation | None:
    """
    Parse a single line for cache annotations.

    *lineno* is the line's 1-based number in its cell, if known; an unknown
    directive's warning points there.

    Returns CacheAnnotation if found, None otherwise.
    """
    match = ANNOTATION_PATTERN.search(line)
    if not match:
        return None

    directive = match.group(1).lower()
    value = match.group(2)

    if directive == "persist":
        return CacheAnnotation(persist=True)
    if directive == "no-cache":
        return CacheAnnotation(no_cache=True)
    if directive == "allow-random":
        return CacheAnnotation(allow_random=True)
    if directive == "cache-fit":
        return CacheAnnotation(cache_fit=True)
    if directive == "no-cache-calls":
        return CacheAnnotation(no_cache_calls=True)
    if directive == "assume-safe":
        return CacheAnnotation(assume_safe=True)
    if directive == "ttl":
        # ``isascii`` as well as ``isdigit``: the latter is True for characters
        # like the superscript two, which ``int()`` then refuses.
        if value is not None and value.isascii() and value.isdigit():
            return CacheAnnotation(ttl=int(value))
        if value is None:
            problem = "`# @cash:ttl` gives no number of seconds"
        else:
            problem = f"`# @cash:ttl={value}` is not a whole number of seconds"
        warn_diagnostic(
            CashCacheIneffectiveWarning,
            "ANNOT-TTL-INVALID",
            f"{problem}, so the annotation was IGNORED and this statement keeps its normal caching with no expiry.",
            "rewrite the value as a bare count of seconds -- `ttl=300` for five "
            "minutes -- with no unit suffix and no decimal point.",
        )
        return None
    if directive not in KNOWN_DIRECTIVES:
        _warn_unknown_directive(directive, lineno)
    return None


def parse_annotations_in_range(source_lines: list[str], start_line: int, end_line: int) -> CacheAnnotation:
    """
    Parse all annotations within a line range (1-indexed, inclusive).

    Also checks consecutive annotation/comment lines before start_line.
    """
    result = CacheAnnotation()

    # Check consecutive lines before the statement (walking backwards)
    # to capture multi-line annotation blocks like:
    # # @cash:persist
    # # @cash:ttl=60
    # statement_here
    check_line = start_line - 2  # -2 for 0-index and -1 line
    while check_line >= 0:
        line = source_lines[check_line]
        stripped = line.strip()

        # Stop if we hit a non-comment, non-empty line
        if stripped and not stripped.startswith("#"):
            break

        # Try to parse annotation from this line
        ann = parse_annotation_line(line, check_line + 1)
        if ann:
            result = result.merge(ann)

        # If it's an empty line, stop looking
        if not stripped:
            break

        check_line -= 1

    # Check all lines within the statement
    for i in range(start_line - 1, min(end_line, len(source_lines))):
        ann = parse_annotation_line(source_lines[i], i + 1)
        if ann:
            result = result.merge(ann)

    return result


def leading_cell_annotation(source_lines: list[str]) -> CacheAnnotation:
    """The cell-scoped directives from the cell's LEADING comment block.

    Only ``no-cache`` and ``no-cache-calls`` propagate from the header to the
    whole cell. That asymmetry is deliberate, and it is about which way each
    directive is safe to be wrong:

    * ``no-cache`` is a SAFETY opt-out. A user writes it because caching this
      cell would be *incorrect* — timestamps, side effects, live values. Applying
      it to only the first statement silently cached statements 2..n of a cell
      explicitly marked do-not-cache, producing exactly the stale
      values the user was trying to prevent. Over-applying it merely costs speed,
      so it fails safe cell-wide.
    * ``no-cache-calls`` is the same shape of opt-out, one level down: call
      interception is on by default, and under default-on the
      placement trap inverts -- someone who needs to disable it for a whole
      cell should not have to annotate every statement in it. Applying it to
      only the first statement would leave statements 2..n intercepted despite
      an explicit cell-wide "don't". Over-applying it merely costs speed, so it
      fails safe cell-wide too.
    * ``persist`` / ``ttl`` are PERFORMANCE hints, and over-applying them is the
      expensive direction: a header ``persist`` spread across a loop that grows a
      frame snapshots every intermediate width — measured at 13x cache
      amplification. They stay statement-scoped, which is also how a
      header ``persist`` above a single statement already reads.
    * ``assume-safe`` is a WAIVER: over-applying it caches statements whose
      side effects nobody audited, so a hit silently skips them. It stays on
      the statement it was written for, as it does in a decorated function.

    Only the header block counts: scanning stops at the first line of real code,
    so a directive further down stays statement-scoped and mid-cell targeting
    keeps working.
    """
    header = CacheAnnotation()
    for lineno, line in enumerate(source_lines, 1):
        stripped = line.strip()
        if not stripped:
            continue  # blank lines don't close the header
        if not stripped.startswith("#"):
            break  # first real code closes the header
        ann = parse_annotation_line(line, lineno)
        if ann:
            header = header.merge(ann)
    # Propagate the safety opt-outs only.
    return CacheAnnotation(no_cache=header.no_cache, no_cache_calls=header.no_cache_calls)


def get_statement_annotations(full_source: str, node: ast.AST) -> CacheAnnotation:
    """
    Get cache annotations that apply to an AST node.

    Handles compound statements by checking all lines within the block, and
    layers the cell's leading-block directives underneath (see
    :func:`leading_cell_annotation`) so a cell-level directive reaches every
    statement, not just the first.

    Args:
        full_source: The complete source code
        node: The AST node to get annotations for

    Returns:
        CacheAnnotation with all applicable directives
    """
    source_lines = full_source.splitlines()

    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return CacheAnnotation()

    start_line = node.lineno
    end_line = node.end_lineno or start_line

    statement_level = parse_annotations_in_range(source_lines, start_line, end_line)
    if is_assume_safe_block(node):
        # ``with cash.assume_safe():`` runs as one statement here, as any
        # ``with`` does, so it waives as the comment anywhere inside it would.
        statement_level = statement_level.merge(CacheAnnotation(assume_safe=True))

    # Cell-level directives reach TOP-LEVEL statements only (``col_offset == 0``).
    # A statement nested in a control body must NOT pick them up implicitly: the
    # control-structure processor decides deliberately what a body inherits and
    # merges it explicitly (resolve_header_annotation -> resolve_statement_
    # annotation). Letting the header leak in here would re-create the bug where
    # a whole-range annotation disabled caching for every sibling in the body.
    if getattr(node, "col_offset", 0) != 0:
        return statement_level

    # Cell-level first, statement-level layered on top, so a statement-specific
    # directive can still refine (and ``no-cache`` still wins over ``persist``
    # via CacheAnnotation.merge).
    return leading_cell_annotation(source_lines).merge(statement_level)


def extract_annotations_for_statements(full_source: str) -> dict[int, CacheAnnotation]:
    """
    Extract all annotations and map them to statement line numbers.

    Returns a dict mapping statement start line -> CacheAnnotation.
    This allows looking up annotations before unparsing AST nodes.
    """
    try:
        tree = ast.parse(full_source)
    except SyntaxError:
        return {}

    annotations = {}
    source_lines = full_source.splitlines()
    # Same cell-level layering as get_statement_annotations — these
    # two must agree, or a directive would apply on one lookup path and not the
    # other.
    cell_level = leading_cell_annotation(source_lines)

    for node in tree.body:
        if hasattr(node, "lineno"):
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", start_line) or start_line
            ann = cell_level.merge(parse_annotations_in_range(source_lines, start_line, end_line))
            if is_assume_safe_block(node):
                ann = ann.merge(CacheAnnotation(assume_safe=True))

            if ann.has_directives():
                annotations[start_line] = ann

    return annotations


#: ``# @cash:assume-safe`` -- a waiver scoped to ONE statement.
#
# ``assume_safe=True`` on the decorator silences the whole function, for good.
# Audit a call today, add an unrelated ``session.post(...)`` next month, and
# nothing says a word: the waiver outlived the audit it was granted for.
# Measured -- a POST added after the fact was detected by the analyzer and
# suppressed by the flag.
#
# A waiver written NEXT TO the statement cannot do that. New code arrives
# unannotated, so it is reported. The scope of the exemption is visible in the
# diff that grants it, which is the property blanket suppression cannot have.
ASSUME_SAFE_RE = re.compile(r"#\s*@cash:\s*assume-safe\b")


def audited_lines(src: str) -> tuple[frozenset[int], bool]:
    """Line numbers waived by ``# @cash:assume-safe``, and the function flag.

    1-based against *src*, the same frame ``PurityIssue.line`` uses -- both
    come from the dedented function source.

    An annotation on its own line waives the statement BELOW it as well as
    itself, because that is how people write ``# noqa`` once the line is long.
    On the ``def`` line it waives the function-scoped findings instead: a read
    of a mutated global is a property of the whole body and carries no line, so
    there is no statement to attach it to.
    """
    lines = src.splitlines()
    marked: set[int] = set()
    for index, line in enumerate(lines, start=1):
        if not ASSUME_SAFE_RE.search(line):
            continue
        marked.add(index)
        if line.strip().startswith("#"):
            marked.add(index + 1)
    function_scope = any(
        lines[i - 1].lstrip().startswith(("def ", "async def ")) for i in marked if 1 <= i <= len(lines)
    )
    return frozenset(marked), function_scope


# ``with cash.assume_safe():`` -- the same waiver for every line of a block.
#
# The comment is prose to Python, so a typo in it waives nothing and says
# nothing. The block is a call: an editor completes it, a type checker checks
# it, and a misspelling fails on the first run. The runtime half, which waives
# the effects the first call is seen to perform, is in `cash.effect_observer`.

#: The name of the waiver, as the spelling-only match reads it.
_WAIVER_NAME = "assume_safe"


def _dotted_parts(expr: ast.expr) -> list[str] | None:
    """``["cash", "assume_safe"]`` for ``cash.assume_safe``; None for anything
    but a name or a chain of attributes on one."""
    parts: list[str] = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if not isinstance(expr, ast.Name):
        return None
    parts.append(expr.id)
    return parts[::-1]


def _waiver_call_parts(expr: ast.expr) -> list[str] | None:
    """The dotted name a no-argument call is made on: ``["cash",
    "assume_safe"]`` for ``cash.assume_safe()``; None for anything else."""
    if not isinstance(expr, ast.Call) or expr.args or expr.keywords:
        return None
    return _dotted_parts(expr.func)


def waiver_items(node: ast.With) -> list[ast.withitem]:
    """The items of *node* spelled as the waiver: ``assume_safe()`` called with
    no arguments, bare or as an attribute (``cash.assume_safe()``,
    ``c.assume_safe()``), bound to no ``as`` name.

    By spelling alone. The cache key uses this, and it has only text to go
    on: a notebook's simulation digests a function from its text before
    anything is imported, and it must agree with the live function.
    """
    found = []
    for item in node.items:
        if item.optional_vars is not None:
            continue
        parts = _waiver_call_parts(item.context_expr)
        if parts is not None and parts[-1] == _WAIVER_NAME:
            found.append(item)
    return found


def is_assume_safe_block(node: ast.AST) -> bool:
    """Is *node* a ``with`` statement whose items include the waiver?

    By spelling (`waiver_items`): the notebook path decides a statement's
    annotation from its text, the same way on a run and in the simulation.
    """
    return isinstance(node, ast.With) and bool(waiver_items(node))


def _names_bound_for(func: Any, tree: ast.AST) -> dict[str, Any]:
    """What names in *func*'s body resolve to besides its globals: its
    closure cells, and the modules and names its own ``import`` lines bind."""
    namespace: dict[str, Any] = {}
    code = getattr(func, "__code__", None)
    for name, cell in zip(getattr(code, "co_freevars", ()) or (), getattr(func, "__closure__", None) or ()):
        try:
            namespace[name] = cell.cell_contents
        except ValueError:  # an empty cell
            continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.partition(".")[0]
                module = sys.modules.get(alias.name if alias.asname else top)
                if module is not None:
                    namespace[alias.asname or top] = module
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            module = sys.modules.get(node.module)
            for alias in node.names:
                if alias.name == "*" or module is None:
                    continue
                value = vars(module).get(alias.name, sys.modules.get(f"{node.module}.{alias.name}"))
                if value is not None:
                    namespace[alias.asname or alias.name] = value
    return namespace


def _resolves_to(parts: list[str], bound: dict[str, Any], globals_: dict[str, Any], target: object) -> bool:
    """Does the dotted name *parts* name *target*, its first part looked up
    in *bound* and then in *globals_*?

    Attributes are read from modules only, and from their ``__dict__``: a
    module's ``__getattr__`` or an object's property never runs for this.
    """
    missing = object()
    obj = bound.get(parts[0], missing)
    if obj is missing:
        obj = globals_.get(parts[0], missing)
    for attr in parts[1:]:
        if not isinstance(obj, types.ModuleType):
            return False
        obj = vars(obj).get(attr, missing)
    return obj is target


def assume_safe_block_lines(tree: ast.AST, func: Any = None) -> frozenset[int]:
    """Line numbers inside a ``with cash.assume_safe():`` block in *tree*.

    The lines of the block's body, in *tree*'s own numbering, which for a
    function is the frame `audited_lines` uses. An item written after the
    waiver on the ``with`` line is inside the block, as Python nests it:
    ``with cash.assume_safe(), open(p, "w") as fh:`` opens the file there.
    One written before it is not.

    With *func*, the call must name ``cash.assume_safe`` in *func*'s
    namespace, however it was imported (``cash.assume_safe``, ``c.`` after
    ``import cash as c``, a ``from cash import assume_safe as waive`` name);
    a function of the user's own that happens to be called ``assume_safe``
    waives nothing. Without *func*, the spelling decides (`waiver_items`).
    """
    candidates: list[tuple[ast.With, list[tuple[int, list[str]]]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With) or not node.body:
            continue
        calls = [
            (index, parts)
            for index, item in enumerate(node.items)
            if item.optional_vars is None and (parts := _waiver_call_parts(item.context_expr)) is not None
        ]
        if calls:
            candidates.append((node, calls))
    if not candidates:
        return frozenset()
    resolves: Callable[[list[str]], bool]
    if func is None:

        def resolves(parts: list[str]) -> bool:
            return parts[-1] == _WAIVER_NAME

    else:
        from ..effect_observer import assume_safe

        bound = _names_bound_for(func, tree)
        globals_ = getattr(func, "__globals__", None) or {}

        def resolves(parts: list[str]) -> bool:
            return _resolves_to(parts, bound, globals_, assume_safe)

    lines: set[int] = set()
    for node, calls in candidates:
        first = next((index for index, parts in calls if resolves(parts)), None)
        if first is None:
            continue
        inner = node.items[first + 1 :]
        start = inner[0].context_expr.lineno if inner else node.body[0].lineno
        lines.update(range(start, (node.end_lineno or node.body[-1].lineno) + 1))
    return frozenset(lines)

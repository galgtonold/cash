from __future__ import annotations

"""Parser for @cash: comment annotations that control per-statement caching."""

import ast
import re
import warnings
from dataclasses import dataclass

from ..exceptions import CashCacheIneffectiveWarning

__all__ = ["CacheAnnotation", "ANNOTATION_PATTERN", "leading_cell_annotation", "parse_annotation_line", "parse_annotations_in_range", "get_statement_annotations", "extract_annotations_for_statements"]

@dataclass
class CacheAnnotation:
    """Represents cache control annotations for a statement."""
    persist: bool = False       # Force disk persistence
    no_cache: bool = False      # Disable caching entirely
    ttl: int | None = None   # Override TTL in seconds
    allow_random: bool = False  # Suppress randomness warnings
    cache_fit: bool = False     # Opt in to caching a bare ``estimator.fit(X, y)``
    cache_calls: bool = False   # No-op: call-interception is the default (CAS-243)
    no_cache_calls: bool = False  # Opt OUT of caching CALLS inside the statement

    def merge(self, other: CacheAnnotation) -> CacheAnnotation:
        """Merge with another annotation (other takes precedence for ttl)."""
        return CacheAnnotation(
            persist=self.persist or other.persist,
            no_cache=self.no_cache or other.no_cache,
            ttl=other.ttl if other.ttl is not None else self.ttl,
            allow_random=self.allow_random or other.allow_random,
            cache_fit=self.cache_fit or other.cache_fit,
            cache_calls=self.cache_calls or other.cache_calls,
            no_cache_calls=self.no_cache_calls or other.no_cache_calls
        )

    def has_directives(self) -> bool:
        """Check if any directives are set."""
        return (self.persist or self.no_cache or self.ttl is not None
                or self.allow_random or self.cache_fit or self.cache_calls
                or self.no_cache_calls)

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
# reads as "cash isn't working" rather than "my annotation was truncated"
# (CAS-249). Capturing the whole token lets the directive handler see ``5m``
# and reject it out loud.
ANNOTATION_PATTERN = re.compile(r'#\s*@cash:\s*([\w-]+)(?:\s*=\s*(\S*))?')

def parse_annotation_line(line: str) -> CacheAnnotation | None:
    """
    Parse a single line for cache annotations.

    Returns CacheAnnotation if found, None otherwise.
    """
    match = ANNOTATION_PATTERN.search(line)
    if not match:
        return None

    directive = match.group(1).lower()
    value = match.group(2)

    if directive == 'persist':
        return CacheAnnotation(persist=True)
    if directive == 'no-cache' or directive == 'nocache':
        return CacheAnnotation(no_cache=True)
    if directive == 'allow-random' or directive == 'allowrandom':
        return CacheAnnotation(allow_random=True)
    if directive == 'cache-fit' or directive == 'cachefit':
        return CacheAnnotation(cache_fit=True)
    if directive == 'cache-calls' or directive == 'cachecalls':
        # Call-interception is the default now (CAS-243); this directive is
        # kept parseable so notebooks written under the opt-in era don't
        # error, but it has no effect.
        return CacheAnnotation(cache_calls=True)
    if directive == 'no-cache-calls' or directive == 'nocachecalls':
        return CacheAnnotation(no_cache_calls=True)
    if directive == 'ttl' and value is not None:
        # ``isascii`` as well as ``isdigit``: the latter is True for characters
        # like the superscript two, which ``int()`` then refuses.
        if value.isascii() and value.isdigit():
            return CacheAnnotation(ttl=int(value))
        warnings.warn(
            f"cash: `# @cash:ttl={value}` is not a whole number of seconds, so "
            f"the annotation was IGNORED and this statement keeps its normal "
            f"caching. ttl has no unit suffix -- write `ttl=300` for five "
            f"minutes, not `ttl=5m`.",
            CashCacheIneffectiveWarning,
            stacklevel=2,
        )
        return None

    return None

def parse_annotations_in_range(
    source_lines: list[str],
    start_line: int,
    end_line: int
) -> CacheAnnotation:
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
        if stripped and not stripped.startswith('#'):
            break

        # Try to parse annotation from this line
        ann = parse_annotation_line(line)
        if ann:
            result = result.merge(ann)

        # If it's an empty line, stop looking
        if not stripped:
            break

        check_line -= 1

    # Check all lines within the statement
    for i in range(start_line - 1, min(end_line, len(source_lines))):
        ann = parse_annotation_line(source_lines[i])
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
      interception is on by default (CAS-243), and under default-on the
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

    Only the header block counts: scanning stops at the first line of real code,
    so a directive further down stays statement-scoped and mid-cell targeting
    keeps working.
    """
    header = CacheAnnotation()
    for line in source_lines:
        stripped = line.strip()
        if not stripped:
            continue                      # blank lines don't close the header
        if not stripped.startswith('#'):
            break                         # first real code closes the header
        ann = parse_annotation_line(line)
        if ann:
            header = header.merge(ann)
    # Propagate the safety opt-outs only.
    return CacheAnnotation(no_cache=header.no_cache, no_cache_calls=header.no_cache_calls)


def get_statement_annotations(
    full_source: str,
    node: ast.AST
) -> CacheAnnotation:
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

    if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
        return CacheAnnotation()

    start_line = node.lineno
    end_line = node.end_lineno or start_line

    statement_level = parse_annotations_in_range(source_lines, start_line, end_line)

    # Cell-level directives reach TOP-LEVEL statements only (``col_offset == 0``).
    # A statement nested in a control body must NOT pick them up implicitly: the
    # control-structure processor decides deliberately what a body inherits and
    # merges it explicitly (resolve_header_annotation -> resolve_statement_
    # annotation). Letting the header leak in here would re-create the bug where
    # a whole-range annotation disabled caching for every sibling in the body.
    if getattr(node, 'col_offset', 0) != 0:
        return statement_level

    # Cell-level first, statement-level layered on top, so a statement-specific
    # directive can still refine (and ``no-cache`` still wins over ``persist``
    # via CacheAnnotation.merge).
    return leading_cell_annotation(source_lines).merge(statement_level)

def extract_annotations_for_statements(
    full_source: str
) -> dict[int, CacheAnnotation]:
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
        if hasattr(node, 'lineno'):
            start_line = node.lineno
            end_line = getattr(node, 'end_lineno', start_line) or start_line
            ann = cell_level.merge(
                parse_annotations_in_range(source_lines, start_line, end_line)
            )

            if ann.has_directives():
                annotations[start_line] = ann

    return annotations


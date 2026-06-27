from __future__ import annotations

"""Parser for @cash: comment annotations that control per-statement caching."""

import ast
import re
from dataclasses import dataclass

__all__ = ["CacheAnnotation", "ANNOTATION_PATTERN", "parse_annotation_line", "parse_annotations_in_range", "get_statement_annotations", "extract_annotations_for_statements"]

@dataclass
class CacheAnnotation:
    """Represents cache control annotations for a statement."""
    persist: bool = False       # Force disk persistence
    no_cache: bool = False      # Disable caching entirely
    ttl: int | None = None   # Override TTL in seconds
    allow_random: bool = False  # Suppress randomness warnings

    def merge(self, other: CacheAnnotation) -> CacheAnnotation:
        """Merge with another annotation (other takes precedence for ttl)."""
        return CacheAnnotation(
            persist=self.persist or other.persist,
            no_cache=self.no_cache or other.no_cache,
            ttl=other.ttl if other.ttl is not None else self.ttl,
            allow_random=self.allow_random or other.allow_random
        )

    def has_directives(self) -> bool:
        """Check if any directives are set."""
        return self.persist or self.no_cache or self.ttl is not None or self.allow_random

# Regex patterns for annotation parsing ([\w-]+ allows hyphens in directive names)
# Whitespace is tolerated after the colon and around ``=`` so both the
# documented ``# @cash:no-cache`` and the natural ``# @cash: no-cache`` parse
# (see test_annotation_with_spaces). Without the ``\s*`` after the colon the
# spaced form was silently ignored -- the statement was cached as normal.
ANNOTATION_PATTERN = re.compile(r'#\s*@cash:\s*([\w-]+)(?:\s*=\s*(\d+))?')

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
    if directive == 'ttl' and value is not None:
        try:
            return CacheAnnotation(ttl=int(value))
        except ValueError:
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

def get_statement_annotations(
    full_source: str,
    node: ast.AST
) -> CacheAnnotation:
    """
    Get cache annotations that apply to an AST node.

    Handles compound statements by checking all lines within the block.

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

    return parse_annotations_in_range(source_lines, start_line, end_line)

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

    for node in tree.body:
        if hasattr(node, 'lineno'):
            start_line = node.lineno
            end_line = getattr(node, 'end_lineno', start_line) or start_line
            ann = parse_annotations_in_range(source_lines, start_line, end_line)

            if ann.has_directives():
                annotations[start_line] = ann

    return annotations


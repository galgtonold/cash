"""The texts of one top-level statement of a cell: the one keyed, the one
the badge shows and the one executed.

The key is always the unparsed form (``statement_code``). The badge shows
the user's own layout (:func:`statement_source`), and a cache miss compiles
that text too, except that a ``def``/``class`` keeps its unparsed form
unless its body carries a waiver (:func:`exec_source_for_node`).
"""

from __future__ import annotations

import ast

from ...analysis.annotations import audited_lines
from ...analysis.code_analyzer import splitlines_like_the_parser, statement_code

__all__ = ["exec_source_for_node", "statement_source", "statement_texts"]


def statement_source(raw_cell: str, node: ast.stmt) -> str | None:
    """The statement's original text, as the badge shows it.

    Also what a cache-miss statement compiles from (``exec_source``) in place
    of the ``ast.unparse`` form, which strips comments; the cache key is
    always the unparsed form. Continuation lines are dedented by the node's
    own ``col_offset``, since ``get_source_segment`` leaves them at their
    absolute indentation.

    A top-level ``def``/``class`` returns ``None``: running it only binds the
    name, and its full body would make the badge very tall. Lifting that
    would also change what EXECUTES, since ``exec_source_for_node`` returns
    this text whenever it is set, and so bypass that function's waiver gate.
    A ``match`` is not excluded: it runs as one unit, so its text is the code
    that ran.

    A single-line statement is sliced out directly: ``get_source_segment``
    re-splits the whole cell on every call (O(statements x cell length)).
    Offsets are UTF-8 byte offsets, as there, and the split follows the
    parser's line breaks (``splitlines_like_the_parser``), not
    ``str.splitlines``'s, or the line index drifts from ``node.lineno``.

    Returns ``None`` when the segment cannot be recovered; the caller falls
    back to the unparsed form. Never raises.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
    try:
        end_lineno = node.end_lineno
        if end_lineno is not None and end_lineno == node.lineno and node.end_col_offset is not None:
            lines = splitlines_like_the_parser(raw_cell)
            segment = lines[node.lineno - 1].encode()[node.col_offset : node.end_col_offset].decode()
        else:
            segment = ast.get_source_segment(raw_cell, node)
    except (IndexError, ValueError, TypeError, AttributeError):  # display only
        return None
    if not segment:
        return None

    indent = getattr(node, "col_offset", 0) or 0
    if indent <= 0:
        return segment

    head, *rest = segment.split("\n")
    return "\n".join([head] + [line[indent:] if line[:indent].isspace() else line for line in rest])


def exec_source_for_node(
    raw_cell: str,
    node: ast.stmt,
    stmt_display: str | None,
) -> str | None:
    """The text to EXECUTE for *node*, where it differs from ``stmt_display``.

    ``stmt_display`` is the text the badge shows. It withholds a top-level
    ``def``/``class`` body (``statement_source``), and the unparsed fallback
    has no comments, so a ``# @cash:assume-safe`` waiver inside a function
    defined in a cell would be invisible to ``inspect.getsource`` and so to
    the purity analyzer. For such a function this recovers the original text,
    decorators and a trailing comment on its last line included.

    Only when the body carries a real waiver, as decided by
    ``annotations.audited_lines`` (the analyzer's own test, so the two
    cannot disagree). Every other ``def``/``class`` must compile from the
    unparsed text on every path: the upstream re-execution path always
    compiles that form, and two texts for one function give it two identity
    hashes, which re-keys every cached call to it. A waived function
    redefined through that path still hashes differently once per session
    (one lost call-cache hit; the values stay correct).

    The recovered text is checked with ``compile()`` (with top-level
    ``await`` allowed, as the async path compiles), since the line-based
    reconstruction can go wrong -- a PEP 614 decorator expression starting
    below its ``@`` line, for one.

    Returns ``stmt_display`` when it is set, else the recovered text, else
    ``None`` (the executor then runs the unparsed code). Never raises.
    """
    if stmt_display is not None:
        return stmt_display
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
    # A pre-filter only: `body` is part of `raw_cell`, so no waiver can be
    # found in it without this text somewhere in the cell.
    if "@cash:" not in raw_cell:
        return None
    try:
        body = ast.get_source_segment(raw_cell, node)
        if not body:
            return None
        lines = splitlines_like_the_parser(raw_cell)
        # The segment starts at the `def`/`class` line, not the decorators.
        decorators = node.decorator_list
        if decorators:
            prefix = "".join(lines[decorators[0].lineno - 1 : node.lineno - 1])
            body = prefix + body
        end_lineno = getattr(node, "end_lineno", None)
        end_col = getattr(node, "end_col_offset", None)
        # ...and ends at `end_col_offset`, before a comment on the last line.
        if end_lineno is not None and end_col is not None and 0 < end_lineno <= len(lines):
            rest = lines[end_lineno - 1].encode()[end_col:].decode()
            trailing = rest.split("\n", 1)[0].rstrip("\r")
            if trailing.strip() == "" or trailing.lstrip().startswith("#"):
                body = body + trailing
        # The function-scope flag is derived from the marked lines, so no
        # marked line means no waiver of either kind.
        waived_lines, _ = audited_lines(body)
        if not waived_lines:
            return None
        # Top-level `await` allowed, as `CodeRunner.run_async` compiles: an
        # `@await get_deco()` decorator is legal there. On the sync path such
        # text fails to compile either way, so the flag decides nothing.
        compile(body, "<cash-recovery-check>", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        return body
    except Exception:  # noqa: BLE001 - execution must never break over this
        return None


def statement_texts(raw_cell: str, node: ast.stmt) -> tuple[str, str | None, str | None] | None:
    """``(code, display_code, exec_source)`` for the top-level *node*, or
    None when it cannot be unparsed.

    ``code`` is the keyed and executed text; the badge shows the user's
    own layout (see `statement_source`).
    """
    # With an expression's trailing ``;`` (``df.head();`` shows no
    # repr): the suppression rides through the cache key AND the
    # execution path (``CodeRunner`` skips the display), so a cached
    # re-run doesn't emit a phantom repr. See ``statement_code``.
    try:
        stmt_code = statement_code(node, raw_cell)
    except (ValueError, TypeError):
        return None
    stmt_display = statement_source(raw_cell, node)
    stmt_exec_source = exec_source_for_node(raw_cell, node, stmt_display)
    return stmt_code, stmt_display, stmt_exec_source

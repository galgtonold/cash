"""``_exec_source_for_node`` -- what a statement is EXECUTED from, which
diverges from ``_statement_source`` (what the badge DISPLAYS) only for a
top-level ``def``/``class`` whose body the purity analyzer recognises as
carrying an ``# @cash:assume-safe`` waiver.

Three defects were found and fixed on the way to this file, and all three
are pinned here so none regresses silently:

* **Dropped decorator.** ``ast.get_source_segment`` anchors to
  ``node.lineno``, which (Python 3.8+) is the ``def``/``class`` KEYWORD
  line, not the decorator -- so a naive recovery for
  ``@c.cache\\ndef audited(n): ...`` returned just ``def audited(n): ...``,
  silently un-decorating the function it executed.
* **Cross-path hash instability.** Recovering a def/class's original text
  UNCONDITIONALLY made a function's compiled source differ depending on
  which internal path (re)created it: the normal cell-executor split loop
  (which now has an original text to offer) vs. the upstream
  checker/restorer's re-execution of an earlier statement (which never
  threads ``display_code``/``exec_source`` and always falls back to the
  unparsed form -- same as a control body or a loop-split iteration).
  Before recovery existed, EVERY path compiled from the same canonical
  ``ast.unparse`` text, so a function's identity hash was stable regardless
  of which path (re)created it. An unconditional recovery breaks that for
  any function whose original text is not byte-identical to its unparsed
  form (measured: a raw string literal survives as ``r'C:\\...'`` from one
  path and as ``'C:\\\\...'`` from the other -- same value, different text,
  different hash), which can silently move the CAS-243 call-cache key for a
  call to that function. Gating recovery on the presence of a ``@cash:``
  directive (the same substring check ``_drop_audited`` itself uses) means
  the overwhelming majority of functions -- the ones with no directive to
  lose -- are completely unaffected by this function, on every path, exactly
  as if it did not exist. See ``tests/test_notebook_integration/
  test_callee_global_capture.py::test_a_same_session_rerun_neither_freezes_nor_accumulates``
  for the end-to-end regression this was caught by.
* **The substring gate itself was a false-positive surface (final
  whole-branch review, finding 1).** ``"@cash:" in body`` fires on any text
  containing that substring, directive or not -- a docstring merely
  mentioning ``@cash:`` in prose, a string literal containing it, or an
  ordinary comment documenting cash's own annotation syntax all matched,
  putting an UNDIRECTED function on the recovery path and re-opening the
  exact cross-path hash instability above for a user who wrote no
  annotation at all. Fixed by gating on
  ``purity_analyzer._audited_lines(body)`` instead -- the same function the
  analyzer itself calls to decide whether a line is waived, so the gate and
  the analyzer agree about what counts as "directed" BY CONSTRUCTION rather
  than by a second, independently-fallible guess. See
  ``test_a_docstring_mentioning_cash_syntax_is_not_a_directive`` and its
  neighbours below for the false-positive shapes this closes, and
  ``_exec_source_for_node``'s own docstring for the full reasoning.
"""
from __future__ import annotations

import ast

from cash.notebook.ipython.cell_executor import _exec_source_for_node


def _node(cell: str, index: int = 0) -> ast.stmt:
    return ast.parse(cell).body[index]


def test_reuses_stmt_display_when_present():
    """The common case: an ordinary (non-def/class) statement's already-
    recovered display text is reused verbatim, not re-derived."""
    cell = "x = 1\n"
    node = _node(cell)
    assert _exec_source_for_node(cell, node, "x = 1") == "x = 1"


def test_none_stmt_display_on_a_non_def_class_node_stays_none():
    """A control body / loop-split iteration / rewritten statement has no
    segment to recover, and this function must not invent one for anything
    other than a def/class."""
    cell = "x = 1\n"
    node = _node(cell)
    assert _exec_source_for_node(cell, node, None) is None


def test_a_function_with_no_directive_is_left_alone():
    """The regression control. A function with nothing for the purity
    analyzer to find must compile from the SAME canonical form no matter
    which internal path (re)creates it -- so this must return None, exactly
    as if the def/class recovery did not exist, even though the segment
    itself IS recoverable."""
    cell = (
        "def compute(v):\n"
        "    return v * 2\n"
    )
    node = _node(cell)
    assert _exec_source_for_node(cell, node, None) is None


def test_a_function_with_the_directive_is_recovered_with_comments():
    cell = (
        "def audited(n):\n"
        "    time.sleep(0.01)  # @cash:assume-safe - audited\n"
        "    return n * 2\n"
    )
    node = _node(cell)
    result = _exec_source_for_node(cell, node, None)
    assert result == cell.rstrip("\n")
    assert "@cash:assume-safe" in result


def test_the_decorator_is_not_dropped():
    """The first defect found in this module: recovering just the def's own
    span (anchored at `node.lineno`, the `def` keyword line since Python 3.8)
    silently drops the decorator, un-caching (and un-purity-checking) the
    function it executes."""
    cell = (
        "@c.cache\n"
        "def audited(n):\n"
        "    x = 1  # @cash:assume-safe\n"
        "    return x\n"
    )
    node = _node(cell)
    result = _exec_source_for_node(cell, node, None)
    assert result == cell.rstrip("\n")
    assert result.startswith("@c.cache\n")


def test_multiple_decorators_and_blank_lines_between_them_survive():
    cell = (
        "@a\n"
        "\n"
        "@b(\n"
        "    1,\n"
        ")\n"
        "def audited(n):\n"
        "    x = 1  # @cash:assume-safe\n"
        "    return x\n"
    )
    node = _node(cell)
    result = _exec_source_for_node(cell, node, None)
    assert result == cell.rstrip("\n")
    # The recovered text must itself compile -- proof the decorator block
    # was captured intact, not merely present as a substring.
    compile(result, "<test>", "exec")


def test_an_async_function_with_the_directive_is_recovered():
    cell = (
        "async def audited(n):\n"
        "    x = 1  # @cash:assume-safe\n"
        "    return x\n"
    )
    node = _node(cell)
    assert _exec_source_for_node(cell, node, None) == cell.rstrip("\n")


def test_a_class_with_the_directive_is_recovered():
    cell = (
        "class Audited:\n"
        "    x = compute()  # @cash:assume-safe\n"
    )
    node = _node(cell)
    assert _exec_source_for_node(cell, node, None) == cell.rstrip("\n")


def test_a_class_with_no_directive_is_left_alone():
    cell = (
        "class Plain:\n"
        "    x = 1\n"
    )
    node = _node(cell)
    assert _exec_source_for_node(cell, node, None) is None


def test_a_cash_directive_on_a_sibling_statement_does_not_leak_in():
    """The gate reads the RECOVERED SEGMENT for this node only. A directive
    on some other top-level statement in the same cell must not make an
    unrelated, undirected function's recovery fire."""
    cell = (
        "def plain(n):\n"
        "    return n\n"
        "x = 1  # @cash:assume-safe\n"
    )
    node = _node(cell, index=0)
    assert _exec_source_for_node(cell, node, None) is None


def test_a_pep614_parenthesised_decorator_returns_none_not_uncompilable_text():
    """FINDING 1 (review round 1), the regression this file exists to guard
    against not recurring. Since PEP 614 (Python 3.9), a decorator's
    EXPRESSION need not start on the same line as the ``@`` --
    ``decorator_list[0].lineno`` is the expression's own line. The manual
    prefix in ``_exec_source_for_node`` (lines from
    ``decorators[0].lineno - 1`` through ``node.lineno - 1``) then drops the
    ``@(`` line and leaves a stray ``)`` -- text that does not compile.

    Before the ``compile()`` sanity check this function returned that
    broken text anyway, and ``_execute_statement`` raised
    ``IndentationError`` trying to parse/compile it -- killing a cell that
    ran fine on base ``479e30e``. This must return ``None`` instead, the
    same as any other unrecoverable segment, so the caller falls back to
    the unparsed form rather than text that cannot compile."""
    cell = (
        "@(\n"
        "    c.cache\n"
        ")\n"
        "def f(n):\n"
        "    return n  # @cash:assume-safe\n"
    )
    node = _node(cell)
    assert _exec_source_for_node(cell, node, None) is None


# --- FINDING 1 (final whole-branch review) -- the substring gate's own
# false-positive surface. `"@cash:" in body` cannot distinguish a real
# directive from the substring merely appearing in prose, a string literal,
# or a comment that documents cash's own syntax rather than invoking it.
# Each case below must return None -- exactly as if the def/class recovery
# did not exist -- and the control that follows proves a REAL directive in
# the same shape is still recovered, so these fail because they are not
# directives, not because nothing is ever recovered any more.

def test_a_docstring_mentioning_cash_syntax_is_not_a_directive():
    """A docstring that merely MENTIONS ``@cash:`` in prose is not a waiver.
    Before the fix, the bare substring check fired here anyway, putting this
    UNDIRECTED function on the recovery path."""
    cell = (
        'def f(n):\n'
        '    """See the @cash: docs."""\n'
        '    return n\n'
    )
    node = _node(cell)
    assert _exec_source_for_node(cell, node, None) is None


def test_a_string_literal_containing_the_substring_is_not_a_directive():
    """A run-time string value, not a comment -- ``inspect.getsource`` and the
    purity analyzer both see it, but it waives nothing."""
    cell = (
        'def f(n):\n'
        '    msg = "@cash: not a directive"\n'
        '    return n\n'
    )
    node = _node(cell)
    assert _exec_source_for_node(cell, node, None) is None


def test_a_prose_comment_mentioning_cash_syntax_is_not_a_directive():
    """An ordinary comment that happens to document cash's own annotation
    syntax, rather than invoke it. The exact shape a maintainer of THIS
    codebase would be most likely to write by accident."""
    cell = (
        'def f(n):\n'
        '    # the @cash: system does this\n'
        '    return n\n'
    )
    node = _node(cell)
    assert _exec_source_for_node(cell, node, None) is None


def test_a_non_waiver_directive_between_the_decorator_and_def_is_not_recovered():
    """A fifth confirmed shape from the review: text landing in the MANUALLY
    RECONSTRUCTED PREFIX (between the decorator and the ``def`` keyword),
    not in ``ast.get_source_segment``'s own span -- a different code path
    than the three cases above, which all sit inside the segment itself.
    ``_audited_lines`` recognises ``assume-safe`` only, so a comment
    documenting a DIFFERENT ``@cash:`` directive here (meaningless in this
    position regardless) must not fire the gate either."""
    cell = (
        "@c.cache\n"
        "# @cash:no-cache - not real here, just documenting the syntax\n"
        "def f(n):\n"
        "    return n\n"
    )
    node = _node(cell)
    assert _exec_source_for_node(cell, node, None) is None


def test_a_real_directive_alongside_the_false_positives_is_still_recovered():
    """The control arm for the four false positives above: same shape, but
    the ``@cash:`` text is now a REAL waiver. Recovery must still fire --
    the fix narrows the gate's TRIGGER, not its ability to recover a
    genuinely directed function."""
    cell = (
        'def f(n):\n'
        '    time.sleep(0.01)  # @cash:assume-safe\n'
        '    return n\n'
    )
    node = _node(cell)
    assert _exec_source_for_node(cell, node, None) == cell.rstrip("\n")

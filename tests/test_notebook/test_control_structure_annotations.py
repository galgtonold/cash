"""Annotation resolution inside control structures (CAS-135 hole 3).

``# @cash:no-cache`` — the documented escape hatch for the deliberate
"unseeded randomness is cached" policy — worked on a top-level statement and was
silently ignored inside a loop body. The parser was never at fault: the
directive binds to the body statement correctly. It was computed in
``cell_executor`` and then dropped, because ``ControlStructureProcessor.process``
had no parameter to receive it.

These tests pin the *scoping* rules, which is where an over-broad fix does its
damage. The governing rule is **annotation granularity follows cache
granularity**: per-statement where the cache is per-statement, whole-unit where
the cache is one unit.

The real-kernel twin is
``tests/test_notebook_integration/test_no_cache_in_control_structures_integration.py``.
"""
from __future__ import annotations

import ast

from cash.notebook.control_structures import helpers as _helpers


def _for_node(cell: str) -> ast.For:
    return next(n for n in ast.parse(cell).body if isinstance(n, ast.For))


class TestBodyStatementScoping:
    """A directive on one body statement must not leak onto its siblings."""

    CELL = (
        "for t in range(3):\n"
        "    # @cash:no-cache\n"
        "    a = draw()\n"
        "    b = expensive(t)\n"
    )

    def test_annotated_body_statement_gets_the_directive(self):
        node = _for_node(self.CELL)
        ann = _helpers.resolve_statement_annotation(self.CELL, node.body[0], None)
        assert ann is not None
        assert ann.no_cache is True

    def test_sibling_body_statement_does_not(self):
        """The discriminating case. Applying the loop's whole-range scan to every
        body statement would satisfy 'the annotated one re-runs' while silently
        destroying the caching of everything beside it."""
        node = _for_node(self.CELL)
        ann = _helpers.resolve_statement_annotation(self.CELL, node.body[1], None)
        assert ann is None or ann.no_cache is False

    def test_loop_header_scan_is_empty_for_a_body_directive(self):
        """The loop itself carries nothing — the directive belongs to a statement
        inside it. This separation is what makes per-statement scoping possible."""
        node = _for_node(self.CELL)
        assert _helpers.resolve_header_annotation(self.CELL, node, None) is None


class TestHeaderScoping:
    """A directive above the ``for`` scopes to the loop, so the body inherits it."""

    CELL = (
        "# @cash:no-cache\n"
        "for t in range(3):\n"
        "    a = draw()\n"
    )

    def test_header_annotation_is_picked_up(self):
        node = _for_node(self.CELL)
        ann = _helpers.resolve_header_annotation(self.CELL, node, None)
        assert ann is not None and ann.no_cache is True

    def test_body_inherits_the_header_directive(self):
        node = _for_node(self.CELL)
        loop_ann = _helpers.resolve_header_annotation(self.CELL, node, None)
        ann = _helpers.resolve_statement_annotation(self.CELL, node.body[0], loop_ann)
        assert ann is not None and ann.no_cache is True

    def test_body_does_not_inherit_it_implicitly(self):
        """Inheritance must be applied explicitly by merging: a body statement's
        own backward walk stops at the non-comment ``for`` line, so it cannot see
        the loop's directive on its own."""
        node = _for_node(self.CELL)
        ann = _helpers.resolve_statement_annotation(self.CELL, node.body[0], None)
        assert ann is None or ann.no_cache is False


class TestUnitScoping:
    """A structure run as ONE cache entry takes a directive from anywhere in it."""

    CELL = (
        "while i < 2:\n"
        "    # @cash:no-cache\n"
        "    w = draw()\n"
        "    i += 1\n"
    )

    def test_unit_annotation_sees_a_body_directive(self):
        node = ast.parse(self.CELL).body[0]
        ann = _helpers.resolve_unit_annotation(self.CELL, node, None)
        assert ann is not None and ann.no_cache is True


class TestInheritanceAndMerging:

    def test_inherited_directive_survives_an_unannotated_statement(self):
        cell = "for t in range(3):\n    a = draw()\n"
        node = _for_node(cell)
        from cash.notebook.annotations import CacheAnnotation
        inherited = CacheAnnotation(no_cache=True)
        ann = _helpers.resolve_statement_annotation(cell, node.body[0], inherited)
        assert ann is not None and ann.no_cache is True

    def test_directives_merge_rather_than_replace(self):
        cell = (
            "# @cash:persist\n"
            "for t in range(3):\n"
            "    # @cash:ttl=60\n"
            "    a = draw()\n"
        )
        node = _for_node(cell)
        loop_ann = _helpers.resolve_header_annotation(cell, node, None)
        ann = _helpers.resolve_statement_annotation(cell, node.body[0], loop_ann)
        assert ann is not None
        assert ann.persist is True   # from the loop header
        assert ann.ttl == 60         # from the body statement

    def test_no_raw_cell_falls_back_to_inherited(self):
        """``raw_cell=None`` is the pre-CAS-135 behaviour, kept so direct callers
        constructing handlers with mock deps keep working."""
        cell = "for t in range(3):\n    # @cash:no-cache\n    a = draw()\n"
        node = _for_node(cell)
        assert _helpers.resolve_statement_annotation(None, node.body[0], None) is None
        assert _helpers.resolve_header_annotation(None, node, None) is None

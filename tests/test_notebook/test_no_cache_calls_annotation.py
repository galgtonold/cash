"""Parsing and merging for the ``# @cash:no-cache-calls`` opt-out (CAS-243).

Call-level caching is on by default now (see ``test_cache_calls_gate_wiring.py``
and ``tests/test_notebook_integration/test_cache_calls_directive.py`` for the
behavioural flip). This is the escape hatch: it disables interception for a
single statement, or -- from a cell's leading comment block, alongside
``no-cache`` -- for every statement in the cell.

``# @cash:cache-calls`` (the old opt-in) is kept parseable so notebooks
written under the opt-in era don't error; ``test_cache_calls_annotation.py``
already covers that it still sets ``cache_calls`` (now inert). What's new
here is ``no_cache_calls`` itself.
"""

import unittest

from cash.notebook.annotations import CacheAnnotation, leading_cell_annotation, parse_annotation_line


class TestNoCacheCallsAnnotation(unittest.TestCase):

    def test_no_cache_calls_is_parsed(self):
        ann = parse_annotation_line("# @cash:no-cache-calls")
        self.assertIsNotNone(ann)
        self.assertTrue(ann.no_cache_calls)
        self.assertFalse(ann.no_cache, "must not disable the statement cache too")

    def test_hyphenless_spelling_is_accepted(self):
        ann = parse_annotation_line("# @cash:nocachecalls")
        self.assertIsNotNone(ann)
        self.assertTrue(ann.no_cache_calls)

    def test_space_after_colon_is_tolerated(self):
        """The spaced form users actually write must not be silently ignored,
        matching the precedent already set for every other directive."""
        ann = parse_annotation_line("# @cash: no-cache-calls")
        self.assertIsNotNone(ann)
        self.assertTrue(ann.no_cache_calls)

    def test_default_is_off(self):
        """Absent the directive, the opt-out is not engaged -- interception
        proceeds (this is the flip: cache_calls used to gate it IN)."""
        self.assertFalse(CacheAnnotation().no_cache_calls)

    def test_has_directives_reports_it(self):
        """A statement carrying only no-cache-calls must not look un-annotated."""
        self.assertTrue(CacheAnnotation(no_cache_calls=True).has_directives())

    def test_merge_is_sticky(self):
        """Merging must OR the flag in from either side, like the other bools."""
        a = parse_annotation_line("# @cash:no-cache-calls")
        b = parse_annotation_line("# @cash:persist")
        merged = a.merge(b)
        self.assertTrue(merged.no_cache_calls)
        self.assertTrue(merged.persist)
        self.assertTrue(b.merge(a).no_cache_calls)

    def test_merge_does_not_cross_wire_no_cache(self):
        """no-cache-calls and no-cache are independent flags -- setting one
        must not silently set the other, in either direction."""
        only_calls = CacheAnnotation(no_cache_calls=True)
        only_stmt = CacheAnnotation(no_cache=True)
        self.assertFalse(only_calls.no_cache)
        self.assertFalse(only_stmt.no_cache_calls)

        both = only_calls.merge(only_stmt)
        self.assertTrue(both.no_cache_calls)
        self.assertTrue(both.no_cache)

    def test_cache_calls_directive_still_parses_as_a_noop(self):
        """Old notebooks carrying the opt-in directive must not error; it
        just does nothing now, and must not imply the opt-out either."""
        ann = parse_annotation_line("# @cash:cache-calls")
        self.assertIsNotNone(ann)
        self.assertTrue(ann.cache_calls)
        self.assertFalse(ann.no_cache_calls)

    def test_propagates_from_the_cell_header(self):
        """Like no-cache, no-cache-calls reaches every top-level statement in
        the cell, not just the first -- under default-on the placement trap
        inverts, so a whole-cell opt-out must not require annotating every
        line."""
        source = [
            "# @cash:no-cache-calls",
            "",
            "a = compute(1)",
            "b = compute(2)",
        ]
        header = leading_cell_annotation(source)
        self.assertTrue(header.no_cache_calls)

    def test_persist_does_not_propagate_from_the_header(self):
        """Positive control: only the two SAFETY opt-outs (no-cache,
        no-cache-calls) propagate cell-wide; performance hints like persist
        stay statement-scoped."""
        source = ["# @cash:persist", "a = compute(1)"]
        header = leading_cell_annotation(source)
        self.assertFalse(header.persist, "persist must stay statement-scoped")
        self.assertFalse(header.no_cache_calls)


if __name__ == "__main__":
    unittest.main()

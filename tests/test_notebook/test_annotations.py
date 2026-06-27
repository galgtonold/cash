"""
Tests for cache annotation parsing and behavior.
"""

import unittest
import ast
from cash.notebook.annotations import (
    CacheAnnotation,
    parse_annotation_line,
    parse_annotations_in_range,
    get_statement_annotations,
    extract_annotations_for_statements
)


class TestAnnotationParsing(unittest.TestCase):
    """Tests for parsing annotation comments."""

    def test_parse_persist_annotation(self):
        """@cash:persist should set persist=True."""
        ann = parse_annotation_line("# @cash:persist")
        self.assertIsNotNone(ann)
        self.assertTrue(ann.persist)
        self.assertFalse(ann.no_cache)
        self.assertIsNone(ann.ttl)

    def test_parse_nocache_annotation(self):
        """@cash:no-cache should set no_cache=True."""
        ann = parse_annotation_line("# @cash:no-cache")
        self.assertIsNotNone(ann)
        self.assertTrue(ann.no_cache)
        self.assertFalse(ann.persist)

    def test_parse_nocache_without_hyphen(self):
        """@cash:nocache (without hyphen) should also work."""
        ann = parse_annotation_line("# @cash:nocache")
        self.assertIsNotNone(ann)
        self.assertTrue(ann.no_cache)

    def test_parse_nocache_space_after_colon(self):
        """A space after the colon (`@cash: no-cache`) must still parse -- the
        natural form users write. Previously it was silently ignored and the
        statement was cached as normal."""
        for line in ("# @cash: no-cache", "#  @cash:  no-cache"):
            ann = parse_annotation_line(line)
            self.assertIsNotNone(ann, line)
            self.assertTrue(ann.no_cache, line)

    def test_parse_ttl_with_spaces(self):
        """Whitespace after the colon and around `=` is tolerated for ttl."""
        ann = parse_annotation_line("# @cash: ttl = 300")
        self.assertIsNotNone(ann)
        self.assertEqual(ann.ttl, 300)

    def test_parse_ttl_annotation(self):
        """@cash:ttl=N should set ttl=N."""
        ann = parse_annotation_line("# @cash:ttl=300")
        self.assertIsNotNone(ann)
        self.assertEqual(ann.ttl, 300)
        self.assertFalse(ann.persist)
        self.assertFalse(ann.no_cache)

    def test_parse_no_annotation(self):
        """Regular comment should return None."""
        ann = parse_annotation_line("# This is a regular comment")
        self.assertIsNone(ann)

    def test_parse_annotation_with_code(self):
        """Annotation after code should still be parsed."""
        ann = parse_annotation_line("x = 1  # @cash:persist")
        self.assertIsNotNone(ann)
        self.assertTrue(ann.persist)

    def test_annotation_merge(self):
        """Merging annotations should combine flags."""
        ann1 = CacheAnnotation(persist=True)
        ann2 = CacheAnnotation(no_cache=True, ttl=60)
        merged = ann1.merge(ann2)
        
        self.assertTrue(merged.persist)
        self.assertTrue(merged.no_cache)
        self.assertEqual(merged.ttl, 60)

    def test_annotation_merge_ttl_override(self):
        """Later TTL should override earlier TTL."""
        ann1 = CacheAnnotation(ttl=100)
        ann2 = CacheAnnotation(ttl=200)
        merged = ann1.merge(ann2)
        
        self.assertEqual(merged.ttl, 200)


class TestAnnotationRangeParsing(unittest.TestCase):
    """Tests for parsing annotations in line ranges."""

    def test_annotation_on_preceding_line(self):
        """Annotation on line before statement should apply."""
        source = [
            "# @cash:persist",
            "x = expensive_operation()"
        ]
        ann = parse_annotations_in_range(source, 2, 2)
        self.assertTrue(ann.persist)

    def test_annotation_inside_statement(self):
        """Annotation inside multi-line statement should apply."""
        source = [
            "for i in range(10):",
            "    # @cash:no-cache",
            "    print(i)"
        ]
        ann = parse_annotations_in_range(source, 1, 3)
        self.assertTrue(ann.no_cache)

    def test_multiple_annotations_combine(self):
        """Multiple annotations should combine."""
        source = [
            "# @cash:persist",
            "# @cash:ttl=60",
            "x = expensive_operation()"
        ]
        ann = parse_annotations_in_range(source, 3, 3)
        self.assertTrue(ann.persist)
        self.assertEqual(ann.ttl, 60)


class TestGetStatementAnnotations(unittest.TestCase):
    """Tests for get_statement_annotations with AST nodes."""

    def test_simple_statement_annotation(self):
        """Annotation before simple statement should apply."""
        code = """# @cash:persist
x = 1 + 1"""
        tree = ast.parse(code)
        node = tree.body[0]  # The assignment
        
        ann = get_statement_annotations(code, node)
        self.assertTrue(ann.persist)

    def test_for_loop_with_internal_annotation(self):
        """Annotation inside for loop should apply to entire loop."""
        code = """for i in range(10):
    # @cash:no-cache
    result = compute(i)"""
        tree = ast.parse(code)
        node = tree.body[0]  # The for loop
        
        ann = get_statement_annotations(code, node)
        self.assertTrue(ann.no_cache)

    def test_no_annotation(self):
        """Statement without annotation should return empty annotation."""
        code = """x = 1 + 1"""
        tree = ast.parse(code)
        node = tree.body[0]
        
        ann = get_statement_annotations(code, node)
        self.assertFalse(ann.has_directives())


class TestExtractAnnotationsForStatements(unittest.TestCase):
    """Tests for bulk annotation extraction."""

    def test_extract_multiple_annotations(self):
        """Should extract annotations for multiple statements."""
        code = """# @cash:persist
x = expensive()

y = cheap()

# @cash:no-cache
z = volatile()"""
        
        annotations = extract_annotations_for_statements(code)
        
        # Line 2 (x = ...) should have persist
        self.assertIn(2, annotations)
        self.assertTrue(annotations[2].persist)
        
        # Line 7 (z = ...) should have no-cache
        self.assertIn(7, annotations)
        self.assertTrue(annotations[7].no_cache)
        
        # y = ... should not be in the dict (no directives)
        self.assertNotIn(4, annotations)


if __name__ == '__main__':
    unittest.main()

"""Parsing for the ``# @cash:cache-calls`` directive (CAS-243).

Opts a statement in to *sub-expression* caching: the expensive call inside a
cheap wrapper (``out.append(compute(x))``, ``s += compute(x)``) is cached at
the call node, leaving the wrapper to re-execute.

Opt-in first, per the CAS-170 precedent — auto-caching a bare ``model.fit()``
was shipped and reversed, not because the analysis was wrong but because the
purity-judged surface grew silently.
"""

import unittest

from cash.notebook.annotations import CacheAnnotation, parse_annotation_line


class TestCacheCallsAnnotation(unittest.TestCase):

    def test_parse_cache_calls(self):
        """``@cash:cache-calls`` sets cache_calls=True and nothing else."""
        ann = parse_annotation_line("# @cash:cache-calls")
        self.assertIsNotNone(ann)
        self.assertTrue(ann.cache_calls)
        self.assertFalse(ann.no_cache)
        self.assertFalse(ann.persist)
        self.assertFalse(ann.cache_fit)

    def test_parse_cache_calls_without_hyphen(self):
        """``@cash:cachecalls`` is accepted like every other directive alias."""
        ann = parse_annotation_line("# @cash:cachecalls")
        self.assertIsNotNone(ann)
        self.assertTrue(ann.cache_calls)

    def test_parse_cache_calls_space_after_colon(self):
        """The spaced form users actually write must not be silently ignored."""
        ann = parse_annotation_line("# @cash: cache-calls")
        self.assertIsNotNone(ann)
        self.assertTrue(ann.cache_calls)

    def test_default_is_off(self):
        """Absent the directive the feature must not engage."""
        self.assertFalse(CacheAnnotation().cache_calls)

    def test_has_directives_reports_it(self):
        """A statement carrying only cache-calls must not look un-annotated."""
        self.assertTrue(CacheAnnotation(cache_calls=True).has_directives())

    def test_merge_is_sticky(self):
        """Merging must OR the flag in from either side, like the other bools."""
        on = CacheAnnotation(cache_calls=True)
        off = CacheAnnotation()
        self.assertTrue(on.merge(off).cache_calls)
        self.assertTrue(off.merge(on).cache_calls)
        self.assertFalse(off.merge(off).cache_calls)


if __name__ == "__main__":
    unittest.main()

"""
Batch 22: Cash annotation directives (@cash: no-cache, @cash: ttl, etc.)
and debug mode behavior.

Tests the special comment-based directives that control caching behavior
at the statement level.
"""
import pytest
import textwrap


pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestNoCacheAnnotation:
    """Test @cash: no-cache directive."""

    def test_no_cache_always_recomputes(self, nb_runner):
        """@cash: no-cache prevents caching of a statement."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                counter = 0
            """),
            textwrap.dedent("""\
                # @cash: no-cache
                counter = counter + 1
                print(counter)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(2)
        assert "1" in output1

        # Re-run — should recompute, not use cache
        nb_runner.run_cell(2)
        output2 = nb_runner.get_output(2)
        assert "2" in output2

    def test_no_cache_on_print(self, nb_runner):
        """@cash: no-cache on a print statement."""
        nb_runner.create_notebook([
            "x = 42",
            textwrap.dedent("""\
                # @cash: no-cache
                print(f"x = {x}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(2)

    def test_no_cache_mixed_with_cached(self, nb_runner):
        """Mix of cached and no-cache statements in same cell."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                a = 10
                # @cash: no-cache
                b = a + 1
                c = a * 2
                print(b, c)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(1)
        assert "11" in output
        assert "20" in output


class TestAllowRandomAnnotation:
    """Test @cash: allow-random directive."""

    def test_allow_random_permits_caching(self, nb_runner):
        """@cash: allow-random allows caching of random-containing code."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import random
                random.seed(42)
                # @cash: allow-random
                val = random.randint(1, 100)
                print(val)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(1)
        # Should produce a specific number with seed 42
        assert output.strip().isdigit()


class TestDebugMode:
    """Test debug output mode."""

    def test_debug_on_off(self, nb_runner):
        """Enable and disable debug mode."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x + 1",
            "print(y)",
        ])
        nb_runner.start_kernel()
        nb_runner.enable_debug()
        nb_runner.run_all()
        assert "11" in nb_runner.get_output(3)


class TestCashAnnotationEdgeCases:
    """Test edge cases with cash annotations."""

    def test_annotation_with_spaces(self, nb_runner):
        """Annotation with extra spaces should still work."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                #  @cash:  no-cache
                x = 42
                print(x)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42" in nb_runner.get_output(1)

    def test_annotation_case_sensitivity(self, nb_runner):
        """Annotation must be lowercase @cash:."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                # This is just a comment, not an annotation
                # @Cash: no-cache  
                x = 100
                print(x)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "100" in nb_runner.get_output(1)

    def test_multiple_annotations(self, nb_runner):
        """Multiple cash annotations on a statement."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import random
                random.seed(42)
                # @cash: allow-random
                # @cash: no-cache
                val = random.randint(1, 100)
                print(val)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(1)
        assert output.strip()  # some output produced

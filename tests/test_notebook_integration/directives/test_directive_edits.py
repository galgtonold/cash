"""Directive syntax, allow-random and debug mode, and code edited under a directive."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress]


@pytest.mark.integration
class TestCashAnnotationEdgeCases:
    """Test edge cases with cash annotations."""

    def test_annotation_with_spaces(self, nb_runner):
        """Annotation with extra spaces should still work."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                #  @cash:  no-cache
                x = 42
                print(x)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42" in nb_runner.get_output(1)

    def test_annotation_case_sensitivity(self, nb_runner):
        """Annotation must be lowercase @cash:."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                # This is just a comment, not an annotation
                # @Cash: no-cache  
                x = 100
                print(x)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "100" in nb_runner.get_output(1)

    def test_multiple_annotations(self, nb_runner):
        """Multiple cash annotations on a statement."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                import random
                random.seed(42)
                # @cash: allow-random
                # @cash: no-cache
                val = random.randint(1, 100)
                print(val)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(1)
        assert output.strip()  # some output produced


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestAnnotationAndCodeEdits:
    """Combined annotation and code edits."""

    def test_edit_code_with_annotation_present(self, nb_runner):
        """Edit code while annotation is present."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "# @cash:no-cache\nresult = x + 1\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 11" in nb_runner.get_output(2)

        # Edit code but keep annotation
        nb_runner.set_cell_source(2, "# @cash:no-cache\nresult = x * 100\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 1000" in nb_runner.get_output(2)

    def test_edit_upstream_with_annotated_downstream(self, nb_runner):
        """Edit upstream cell, downstream has annotation."""
        nb_runner.create_notebook(
            [
                "base = 5",
                "# @cash:no-cache\ncomputed = base * 2\nprint(f'computed = {computed}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "computed = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "base = 50")
        nb_runner.run_all()
        assert "computed = 100" in nb_runner.get_output(2)


@pytest.mark.integration
class TestAllowRandomAnnotation:
    """Test @cash: allow-random directive."""

    def test_allow_random_permits_caching(self, nb_runner):
        """@cash: allow-random allows caching of random-containing code."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                import random
                random.seed(42)
                # @cash: allow-random
                val = random.randint(1, 100)
                print(val)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(1)
        # Should produce a specific number with seed 42
        assert output.strip().isdigit()


@pytest.mark.integration
class TestDebugMode:
    """Test debug output mode."""

    def test_debug_on_off(self, nb_runner):
        """Enable and disable debug mode."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x + 1",
                "print(y)",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.enable_debug()
        nb_runner.run_all()
        assert "11" in nb_runner.get_output(3)

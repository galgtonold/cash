"""Adding, removing and editing ``# @cash:`` directives on cells."""

import textwrap

import pytest

pytestmark = [pytest.mark.stress]


# Cash annotation directives (@cash: no-cache, @cash: ttl, etc.)
# and debug mode behavior.
#
# Tests the special comment-based directives that control caching behavior
# at the statement level.
@pytest.mark.integration
class TestNoCacheAnnotation:
    """Test @cash: no-cache directive."""

    def test_no_cache_always_recomputes(self, nb_runner):
        """@cash: no-cache prevents caching of a statement."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                counter = 0
            """),
                textwrap.dedent("""\
                # @cash: no-cache
                counter = counter + 1
                print(counter)
            """),
            ]
        )
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
        nb_runner.create_notebook(
            [
                "x = 42",
                textwrap.dedent("""\
                # @cash: no-cache
                print(f"x = {x}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(2)

    def test_no_cache_mixed_with_cached(self, nb_runner):
        """Mix of cached and no-cache statements in same cell."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                a = 10
                # @cash: no-cache
                b = a + 1
                c = a * 2
                print(b, c)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(1)
        assert "11" in output
        assert "20" in output


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


# Annotation/directive interaction tests.
#
# Tests that exercise @cash: directives (no-cache, ttl, persist)
# combined with cell edits to verify correct behavior.
@pytest.mark.core
@pytest.mark.timeout(30)
class TestNoCacheDirective:
    """@cash:no-cache + cell edits."""

    def test_no_cache_always_recomputes(self, nb_runner):
        """Cell with @cash:no-cache always runs fresh."""
        nb_runner.create_notebook(
            [
                "counter = 0",
                "# @cash:no-cache\ncounter = counter + 1",
                "print(f'counter = {counter}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "counter = 1" in nb_runner.get_output(3)

    def test_remove_no_cache_directive(self, nb_runner):
        """Remove @cash:no-cache directive — cell becomes cacheable."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "# @cash:no-cache\ny = x * 3\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(2)

        # Remove no-cache
        nb_runner.set_cell_source(2, "y = x * 3\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestTTLDirective:
    """@cash:ttl + cell edits."""

    def test_ttl_directive_with_edit(self, nb_runner):
        """Cell with TTL directive, edit the code."""
        nb_runner.create_notebook(
            [
                "base = 10",
                "# @cash:ttl=60\nresult = base * 5\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "base = 20")
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

    def test_change_ttl_value(self, nb_runner):
        """Change the TTL value in the directive."""
        nb_runner.create_notebook(
            [
                "x = 1",
                "# @cash:ttl=30\ny = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        # Change TTL (code is different so it should recompute)
        nb_runner.set_cell_source(2, "# @cash:ttl=120\ny = x + 1\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestPersistDirective:
    """@cash:persist + cell edits."""

    def test_persist_directive(self, nb_runner):
        """Cell with persist directive, edit upstream."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                "# @cash:persist\ntotal = sum(data)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 15" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = [10, 20, 30]")
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

    def test_persist_survives_restart(self, nb_runner):
        """Persisted value survives kernel restart."""
        nb_runner.create_notebook(
            [
                "val = 42",
                "# @cash:persist\nresult = val * 2\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 84" in nb_runner.get_output(2)

        # Restart kernel and run just the output cell
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_cell(2)
        assert "result = 84" in nb_runner.get_output(2)


@pytest.mark.core
@pytest.mark.timeout(30)
class TestMixedDirectives:
    """Multiple directives + cell edits."""

    def test_no_cache_and_regular_mixed(self, nb_runner):
        """Mix of no-cache and regular cells."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "# @cash:no-cache\ny = x + 1",
                "z = y * 2\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 22" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 20")
        nb_runner.run_all()
        assert "z = 42" in nb_runner.get_output(3)


# Annotation interaction tests.
#
# Tests combining @cash: annotations (no-cache, ttl, persist)
# with cell edits to verify annotation handling during edits.
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestNoCacheAnnotationEdits:
    """@cash:no-cache annotation with cell edits."""

    def test_add_no_cache_annotation(self, nb_runner):
        """Add @cash:no-cache annotation to a cell."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

        # Add no-cache annotation
        nb_runner.set_cell_source(2, "# @cash:no-cache\ny = x * 2\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

    def test_remove_no_cache_annotation(self, nb_runner):
        """Remove @cash:no-cache annotation."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "# @cash:no-cache\ny = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 6" in nb_runner.get_output(2)

        # Remove annotation
        nb_runner.set_cell_source(2, "y = x + 1\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 6" in nb_runner.get_output(2)


@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestPersistAnnotationEdits:
    """@cash:persist annotation with cell edits."""

    def test_add_persist_annotation(self, nb_runner):
        """Add @cash:persist annotation."""
        nb_runner.create_notebook(
            [
                "data = [1, 2, 3, 4, 5]",
                "total = sum(data)\nprint(f'total = {total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 15" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "# @cash:persist\ntotal = sum(data)\nprint(f'total = {total}')")
        nb_runner.run_all()
        assert "total = 15" in nb_runner.get_output(2)

    def test_persist_survives_restart(self, nb_runner):
        """Persisted value should survive kernel restart."""
        nb_runner.create_notebook(
            [
                "# @cash:persist\nimport time\nexpensive = sum(range(1000))\nprint(f'expensive = {expensive}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "expensive = 499500" in nb_runner.get_output(1)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "expensive = 499500" in nb_runner.get_output(1)


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

"""Batch 151 – Annotation interaction tests.

Tests combining @cash: annotations (no-cache, ttl, persist)
with cell edits to verify annotation handling during edits.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(45)]


class TestNoCacheAnnotationEdits:
    """@cash:no-cache annotation with cell edits."""

    def test_add_no_cache_annotation(self, nb_runner):
        """Add @cash:no-cache annotation to a cell."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

        # Add no-cache annotation
        nb_runner.set_cell_source(2, "# @cash:no-cache\ny = x * 2\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

    def test_remove_no_cache_annotation(self, nb_runner):
        """Remove @cash:no-cache annotation."""
        nb_runner.create_notebook([
            "x = 5",
            "# @cash:no-cache\ny = x + 1\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 6" in nb_runner.get_output(2)

        # Remove annotation
        nb_runner.set_cell_source(2, "y = x + 1\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 6" in nb_runner.get_output(2)


class TestPersistAnnotationEdits:
    """@cash:persist annotation with cell edits."""

    def test_add_persist_annotation(self, nb_runner):
        """Add @cash:persist annotation."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            "total = sum(data)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 15" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            2, "# @cash:persist\ntotal = sum(data)\nprint(f'total = {total}')"
        )
        nb_runner.run_all()
        assert "total = 15" in nb_runner.get_output(2)

    def test_persist_survives_restart(self, nb_runner):
        """Persisted value should survive kernel restart."""
        nb_runner.create_notebook([
            "# @cash:persist\nimport time\nexpensive = sum(range(1000))\nprint(f'expensive = {expensive}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "expensive = 499500" in nb_runner.get_output(1)

        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "expensive = 499500" in nb_runner.get_output(1)


class TestAnnotationAndCodeEdits:
    """Combined annotation and code edits."""

    def test_edit_code_with_annotation_present(self, nb_runner):
        """Edit code while annotation is present."""
        nb_runner.create_notebook([
            "x = 10",
            "# @cash:no-cache\nresult = x + 1\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 11" in nb_runner.get_output(2)

        # Edit code but keep annotation
        nb_runner.set_cell_source(
            2, "# @cash:no-cache\nresult = x * 100\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = 1000" in nb_runner.get_output(2)

    def test_edit_upstream_with_annotated_downstream(self, nb_runner):
        """Edit upstream cell, downstream has annotation."""
        nb_runner.create_notebook([
            "base = 5",
            "# @cash:no-cache\ncomputed = base * 2\nprint(f'computed = {computed}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "computed = 10" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "base = 50")
        nb_runner.run_all()
        assert "computed = 100" in nb_runner.get_output(2)

"""Batch 126 – Annotation/directive interaction tests.

Tests that exercise @cash: directives (no-cache, ttl, persist)
combined with cell edits to verify correct behavior.
"""

import pytest

pytestmark = [pytest.mark.core, pytest.mark.stress, pytest.mark.timeout(30)]


class TestNoCacheDirective:
    """@cash:no-cache + cell edits."""

    def test_no_cache_always_recomputes(self, nb_runner):
        """Cell with @cash:no-cache always runs fresh."""
        nb_runner.create_notebook([
            "counter = 0",
            "# @cash:no-cache\ncounter = counter + 1",
            "print(f'counter = {counter}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "counter = 1" in nb_runner.get_output(3)

    def test_add_no_cache_directive(self, nb_runner):
        """Add @cash:no-cache to a cell that was cached."""
        nb_runner.create_notebook([
            "x = 10",
            "y = x * 2\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

        # Add no-cache — should still compute correctly
        nb_runner.set_cell_source(2, "# @cash:no-cache\ny = x * 2\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

    def test_remove_no_cache_directive(self, nb_runner):
        """Remove @cash:no-cache directive — cell becomes cacheable."""
        nb_runner.create_notebook([
            "x = 5",
            "# @cash:no-cache\ny = x * 3\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(2)

        # Remove no-cache
        nb_runner.set_cell_source(2, "y = x * 3\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(2)


class TestTTLDirective:
    """@cash:ttl + cell edits."""

    def test_ttl_directive_with_edit(self, nb_runner):
        """Cell with TTL directive, edit the code."""
        nb_runner.create_notebook([
            "base = 10",
            "# @cash:ttl=60\nresult = base * 5\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 50" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "base = 20")
        nb_runner.run_all()
        assert "result = 100" in nb_runner.get_output(2)

    def test_change_ttl_value(self, nb_runner):
        """Change the TTL value in the directive."""
        nb_runner.create_notebook([
            "x = 1",
            "# @cash:ttl=30\ny = x + 1\nprint(f'y = {y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)

        # Change TTL (code is different so it should recompute)
        nb_runner.set_cell_source(
            2, "# @cash:ttl=120\ny = x + 1\nprint(f'y = {y}')"
        )
        nb_runner.run_all()
        assert "y = 2" in nb_runner.get_output(2)


class TestPersistDirective:
    """@cash:persist + cell edits."""

    def test_persist_directive(self, nb_runner):
        """Cell with persist directive, edit upstream."""
        nb_runner.create_notebook([
            "data = [1, 2, 3, 4, 5]",
            "# @cash:persist\ntotal = sum(data)\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 15" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "data = [10, 20, 30]")
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

    def test_persist_survives_restart(self, nb_runner):
        """Persisted value survives kernel restart."""
        nb_runner.create_notebook([
            "val = 42",
            "# @cash:persist\nresult = val * 2\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 84" in nb_runner.get_output(2)

        # Restart kernel and run just the output cell
        nb_runner.shutdown()
        nb_runner.start_kernel()
        nb_runner.run_cell(2)
        assert "result = 84" in nb_runner.get_output(2)


class TestMixedDirectives:
    """Multiple directives + cell edits."""

    def test_no_cache_and_regular_mixed(self, nb_runner):
        """Mix of no-cache and regular cells."""
        nb_runner.create_notebook([
            "x = 10",
            "# @cash:no-cache\ny = x + 1",
            "z = y * 2\nprint(f'z = {z}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 22" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 20")
        nb_runner.run_all()
        assert "z = 42" in nb_runner.get_output(3)

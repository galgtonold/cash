"""``# @cash: persist`` and ``# @cash: ttl`` on cells, edited and across a restart."""

import pytest

pytestmark = [pytest.mark.stress]


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

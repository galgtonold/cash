"""Batch 408: string partition and rsplit operations."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringPartitionRsplit:
    def test_partition(self, nb_runner):
        nb_runner.create_notebook([
            "text = 'user@domain.com'",
            "before, sep, after = text.partition('@')\nprint(f'before={before} sep={sep} after={after}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "before=user" in out
        assert "sep=@" in out
        assert "after=domain.com" in out

    def test_rsplit(self, nb_runner):
        nb_runner.create_notebook([
            "path = 'a/b/c/d/e'",
            "parts = path.rsplit('/', maxsplit=2)\nprint(f'parts={parts}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "parts=['a/b/c', 'd', 'e']" in nb_runner.get_output(2)

    def test_partition_edit(self, nb_runner):
        nb_runner.create_notebook([
            "line = 'key=value'",
            "k, _, v = line.partition('=')\nprint(f'k={k} v={v}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "k=key" in nb_runner.get_output(2)
        assert "v=value" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "line = 'name:Alice'")
        nb_runner.set_cell_source(2, "k, _, v = line.partition(':')\nprint(f'k={k} v={v}')")
        nb_runner.run_all()
        assert "k=name" in nb_runner.get_output(2)
        assert "v=Alice" in nb_runner.get_output(2)

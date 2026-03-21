"""
Interaction test: string partition and rpartition methods.
Tests str.partition, str.rpartition for splitting around separators,
cross-cell string processing pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringPartitionRpartition:
    """Test string partition and rpartition across cells."""

    def test_partition_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: partition from left
            "url = 'https://example.com/path/to/resource'\nscheme, sep, rest = url.partition('://')\nprint(f'scheme={scheme}')\nprint(f'rest={rest}')",
            # Cell 2: rpartition from right
            "path_part, slash, filename = rest.rpartition('/')\nprint(f'path_part={path_part}')\nprint(f'filename={filename}')",
            # Cell 3: combine results
            "full_path = f'{scheme}://{path_part}'\nprint(f'full_path={full_path}')\nprint(f'file={filename}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "scheme=https" in out1
        assert "rest=example.com/path/to/resource" in out1
        out2 = nb_runner.get_output(2)
        assert "path_part=example.com/path/to" in out2
        assert "filename=resource" in out2
        out3 = nb_runner.get_output(3)
        assert "full_path=https://example.com/path/to" in out3
        assert "file=resource" in out3

    def test_partition_edit(self, nb_runner):
        nb_runner.create_notebook([
            "email = 'user@example.com'\nlocal, at, domain = email.partition('@')\nprint(f'local={local}')\nprint(f'domain={domain}')",
            "result = f'{local} at {domain}'\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=user at example.com" in nb_runner.get_output(2)

        # Edit email
        nb_runner.set_cell_source(1, "email = 'admin@company.org'\nlocal, at, domain = email.partition('@')\nprint(f'local={local}')\nprint(f'domain={domain}')")
        nb_runner.run_cells([1, 2])
        assert "result=admin at company.org" in nb_runner.get_output(2)

    def test_partition_cache(self, nb_runner):
        nb_runner.create_notebook([
            "line = 'key=value=extra'\nk, eq, v = line.partition('=')\nprint(f'key={k}')\nprint(f'value={v}')",
            "info = f'{k} -> {v}'\nprint(f'info={info}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "key=key" in nb_runner.get_output(1)
        assert "value=value=extra" in nb_runner.get_output(1)
        assert "info=key -> value=extra" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "info=key -> value=extra" in nb_runner.get_output(2)

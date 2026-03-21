"""
Interaction test: string methods partition and rpartition.
Tests str.partition and rpartition for splitting around separators,
with cross-cell parsing pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestPartitionRpartition:
    """Test str.partition and rpartition across cells."""

    def test_partition_operations(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: basic partition
            "url = 'https://example.com:8080/path/to/resource'\nscheme, _, rest = url.partition('://')\nhost_port, _, path = rest.partition('/')\nprint(f'scheme={scheme}')\nprint(f'host_port={host_port}')\nprint(f'path={path}')",
            # Cell 2: rpartition for last separator
            "filepath = 'home/user/docs/report.final.pdf'\ndir_part, _, filename = filepath.rpartition('/')\nbase, _, ext = filename.rpartition('.')\nprint(f'dir={dir_part}')\nprint(f'base={base}')\nprint(f'ext={ext}')",
            # Cell 3: combine parsed info
            "info = f'{scheme}://{host_port}/{base}.{ext}'\nprint(f'info={info}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "scheme=https" in out1
        assert "host_port=example.com:8080" in out1
        assert "path=path/to/resource" in out1
        out2 = nb_runner.get_output(2)
        assert "dir=home/user/docs" in out2
        assert "base=report.final" in out2
        assert "ext=pdf" in out2
        out3 = nb_runner.get_output(3)
        assert "info=https://example.com:8080/report.final.pdf" in out3

    def test_partition_edit(self, nb_runner):
        nb_runner.create_notebook([
            "email = 'user@example.com'\nlocal, _, domain = email.partition('@')\nprint(f'local={local}')\nprint(f'domain={domain}')",
            "tld = domain.rpartition('.')[2]\nprint(f'tld={tld}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "local=user" in nb_runner.get_output(1)
        assert "tld=com" in nb_runner.get_output(2)

        # Edit email
        nb_runner.set_cell_source(1, "email = 'admin@mail.example.co.uk'\nlocal, _, domain = email.partition('@')\nprint(f'local={local}')\nprint(f'domain={domain}')")
        nb_runner.run_cells([1, 2])
        assert "local=admin" in nb_runner.get_output(1)
        assert "tld=uk" in nb_runner.get_output(2)

    def test_partition_cache(self, nb_runner):
        nb_runner.create_notebook([
            "kv = 'name=John Doe'\nkey, _, val = kv.partition('=')\nprint(f'key={key}')\nprint(f'val={val}')",
            "upper_key = key.upper()\nprint(f'upper={upper_key}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "key=name" in nb_runner.get_output(1)
        assert "upper=NAME" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "upper=NAME" in nb_runner.get_output(2)

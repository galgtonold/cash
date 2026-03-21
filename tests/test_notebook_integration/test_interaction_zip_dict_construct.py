"""
Batch 315: zip-to-dict construction patterns with caching.
Tests zip pairing, dict construction, key/value extraction, and edit propagation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.stress, pytest.mark.timeout(90)]


class TestZipDictConstruct:
    """Test zip-based dict construction caching."""

    def test_zip_dict_basic(self, nb_runner):
        """Zip two lists into a dict, verify caching."""
        nb_runner.create_notebook([
            "keys = ['a', 'b', 'c']",
            "values = [1, 2, 3]",
            "mapping = dict(zip(keys, values))",
            "result = ', '.join(f'{k}={v}' for k, v in sorted(mapping.items()))\nprint(result)",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "a=1" in out
        assert "b=2" in out
        assert "c=3" in out

        # Re-run: should be cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "a=1" in out2

    def test_zip_dict_edit_keys(self, nb_runner):
        """Edit keys list, verify dict reconstruction."""
        nb_runner.create_notebook([
            "keys = ['x', 'y']",
            "vals = [10, 20]",
            "d = dict(zip(keys, vals))",
            "total = sum(d.values())\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=30" in out

        nb_runner.set_cell_source(1, "keys = ['x', 'y', 'z']")
        nb_runner.set_cell_source(2, "vals = [10, 20, 30]")
        nb_runner.run_all()
        out2 = nb_runner.get_output(4)
        assert "total=60" in out2

    def test_zip_enumerate_pattern(self, nb_runner):
        """Zip with enumerate for indexed pairs."""
        nb_runner.create_notebook([
            "items = ['apple', 'banana', 'cherry']",
            "indexed = dict(enumerate(items))",
            "lines = [f'{i}: {v}' for i, v in sorted(indexed.items())]\nprint('\\n'.join(lines))",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "0: apple" in out
        assert "2: cherry" in out

        # Re-run cached
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "0: apple" in out2

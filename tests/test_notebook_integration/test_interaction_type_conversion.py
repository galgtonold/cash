"""Batch 186 – Type conversion / coercion chain interaction tests.

Tests editing type conversions (int→str→float, etc.),
serialization round-trips, and format changes.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestTypeConversionEdits:
    """Editing type conversion chains."""

    def test_edit_conversion_chain(self, nb_runner):
        """Edit a type conversion chain."""
        nb_runner.create_notebook([
            "raw = '42.5'  # type conversion source",
            "value = int(float(raw))\nprint(f'value = {value}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "value = 42" in nb_runner.get_output(2)

        # Change chain to round instead of truncate
        nb_runner.set_cell_source(
            2, "value = round(float(raw))\nprint(f'value = {value}')"
        )
        nb_runner.run_all()
        assert "value = 42" in nb_runner.get_output(2)

        # Change source value
        nb_runner.set_cell_source(1, "raw = '42.7'  # type conversion source v2")
        nb_runner.run_all()
        assert "value = 43" in nb_runner.get_output(2)

    def test_edit_format_conversion(self, nb_runner):
        """Edit between different format representations."""
        nb_runner.create_notebook([
            "number = 255  # format source",
            "formatted = hex(number)\nprint(f'formatted = {formatted}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "formatted = 0xff" in nb_runner.get_output(2)

        # Change to binary
        nb_runner.set_cell_source(
            2, "formatted = bin(number)\nprint(f'formatted = {formatted}')"
        )
        nb_runner.run_all()
        assert "formatted = 0b11111111" in nb_runner.get_output(2)

        # Change to octal
        nb_runner.set_cell_source(
            2, "formatted = oct(number)\nprint(f'formatted = {formatted}')"
        )
        nb_runner.run_all()
        assert "formatted = 0o377" in nb_runner.get_output(2)


class TestSerializationEdits:
    """Editing serialization round-trips."""

    def test_json_roundtrip_edit(self, nb_runner):
        """Edit data in a JSON serialization round-trip."""
        nb_runner.create_notebook([
            "import json",
            "data = {'name': 'Alice', 'age': 30}",
            "serialized = json.dumps(data)\nrestored = json.loads(serialized)\nprint(f'name={restored[\"name\"]} age={restored[\"age\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name=Alice age=30" in nb_runner.get_output(3)

        # Change data
        nb_runner.set_cell_source(2, "data = {'name': 'Bob', 'age': 25}")
        nb_runner.run_all()
        assert "name=Bob age=25" in nb_runner.get_output(3)

    def test_csv_roundtrip_edit(self, nb_runner, tmp_path):
        """Edit data in a CSV write/read round-trip."""
        fpath = str(tmp_path / "roundtrip.csv").replace("\\", "/")
        nb_runner.create_notebook([
            f"import csv\nfpath = '{fpath}'",
            "with open(fpath, 'w', newline='') as f:\n    w = csv.writer(f)\n    w.writerow(['a', 'b'])\n    w.writerow([1, 2])",
            "with open(fpath) as f:\n    reader = csv.reader(f)\n    rows = list(reader)\nprint(f'rows = {rows}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "['a', 'b']" in out
        assert "['1', '2']" in out

        # Change written data
        nb_runner.set_cell_source(
            2,
            "with open(fpath, 'w', newline='') as f:\n    w = csv.writer(f)\n    w.writerow(['x', 'y', 'z'])\n    w.writerow([10, 20, 30])",
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "['x', 'y', 'z']" in out2
        assert "['10', '20', '30']" in out2

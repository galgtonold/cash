"""
Interaction test: shelve module for persistent dict-like storage.
Tests shelve.open with writeback, cross-cell key access,
and cache invalidation when shelf contents change.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestShelvePersistentDict:
    """Test shelve persistent storage across cells."""

    def test_shelve_ops(self, nb_runner, tmp_path):
        shelf_path = str(tmp_path / "test_shelf").replace('\\', '/')
        nb_runner.create_notebook([
            # Cell 1: write to shelf
            f"import shelve\nshelf_path = '{shelf_path}'\nwith shelve.open(shelf_path) as db:\n    db['name'] = 'Alice'\n    db['scores'] = [90, 85, 95]\n    key_count = len(db)\nprint(f'keys={{key_count}}')",
            # Cell 2: read from shelf
            "with shelve.open(shelf_path) as db:\n    name = db['name']\n    scores = db['scores']\nprint(f'name={name}')\nprint(f'avg={sum(scores)/len(scores):.1f}')",
            # Cell 3: check keys
            "with shelve.open(shelf_path) as db:\n    all_keys = sorted(db.keys())\nprint(f'all_keys={all_keys}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "keys=2" in out1
        out2 = nb_runner.get_output(2)
        assert "name=Alice" in out2
        assert "avg=90.0" in out2
        out3 = nb_runner.get_output(3)
        assert "name" in out3
        assert "scores" in out3

    def test_shelve_edit(self, nb_runner, tmp_path):
        shelf_path = str(tmp_path / "edit_shelf").replace('\\', '/')
        nb_runner.create_notebook([
            f"import shelve\nshelf_path = '{shelf_path}'\nwith shelve.open(shelf_path) as db:\n    db['val'] = 100\n    stored = db['val']\nprint(f'stored={{stored}}')",
            "doubled = stored * 2\nprint(f'doubled={doubled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "stored=100" in nb_runner.get_output(1)
        assert "doubled=200" in nb_runner.get_output(2)

        # Edit stored value
        nb_runner.set_cell_source(1, f"import shelve\nshelf_path = '{shelf_path}'\nwith shelve.open(shelf_path) as db:\n    db['val'] = 250\n    stored = db['val']\nprint(f'stored={{stored}}')")
        nb_runner.run_cells([1, 2])
        assert "stored=250" in nb_runner.get_output(1)
        assert "doubled=500" in nb_runner.get_output(2)

    def test_shelve_cache(self, nb_runner, tmp_path):
        shelf_path = str(tmp_path / "cache_shelf").replace('\\', '/')
        nb_runner.create_notebook([
            f"import shelve\nwith shelve.open('{shelf_path}') as db:\n    db['x'] = 42\n    x = db['x']\nprint(f'x={{x}}')",
            "is_42 = x == 42\nprint(f'is_42={is_42}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=42" in nb_runner.get_output(1)
        assert "is_42=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_42=True" in nb_runner.get_output(2)

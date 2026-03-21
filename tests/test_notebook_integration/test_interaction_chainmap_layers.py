"""
Interaction test: ChainMap for layered dictionaries.
Tests collections.ChainMap with multiple layers, new_child,
parent traversal, and cross-cell config overlay patterns.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestChainmapLayers:
    """Test ChainMap layered lookup across cells."""

    def test_chainmap_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: create chain map
            "from collections import ChainMap\ndefaults = {'color': 'red', 'size': 'medium', 'style': 'bold'}\nuser_prefs = {'color': 'blue', 'font': 'arial'}\nconfig = ChainMap(user_prefs, defaults)\nprint(f'color={config[\"color\"]}')\nprint(f'size={config[\"size\"]}')\nprint(f'font={config[\"font\"]}')",
            # Cell 2: new_child for override
            "session = config.new_child({'color': 'green', 'zoom': 150})\nprint(f'session_color={session[\"color\"]}')\nprint(f'session_size={session[\"size\"]}')\nprint(f'original_color={config[\"color\"]}')",
            # Cell 3: list all unique keys
            "all_keys = sorted(set(session))\nprint(f'all_keys={all_keys}')\nprint(f'key_count={len(all_keys)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "color=blue" in out1
        assert "size=medium" in out1
        assert "font=arial" in out1
        out2 = nb_runner.get_output(2)
        assert "session_color=green" in out2
        assert "session_size=medium" in out2
        assert "original_color=blue" in out2
        out3 = nb_runner.get_output(3)
        assert "key_count=5" in out3

    def test_chainmap_edit(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import ChainMap\nbase = {'a': 1, 'b': 2}\noverlay = {'b': 20, 'c': 30}\ncm = ChainMap(overlay, base)\nprint(f'a={cm[\"a\"]} b={cm[\"b\"]} c={cm[\"c\"]}')",
            "total = sum(cm.values())\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "a=1 b=20 c=30" in nb_runner.get_output(1)
        # ChainMap.values() returns values from first occurrence of each key
        # Keys: a=1, b=20, c=30 => total=51
        assert "total=51" in nb_runner.get_output(2)

        # Edit overlay
        nb_runner.set_cell_source(1, "from collections import ChainMap\nbase = {'a': 1, 'b': 2}\noverlay = {'b': 200, 'c': 300}\ncm = ChainMap(overlay, base)\nprint(f'a={cm[\"a\"]} b={cm[\"b\"]} c={cm[\"c\"]}')")
        nb_runner.run_cells([1, 2])
        assert "a=1 b=200 c=300" in nb_runner.get_output(1)
        assert "total=501" in nb_runner.get_output(2)

    def test_chainmap_cache(self, nb_runner):
        nb_runner.create_notebook([
            "from collections import ChainMap\nenv = ChainMap({'PATH': '/usr/bin'}, {'HOME': '/home/user', 'PATH': '/bin'})\npath = env['PATH']\nprint(f'path={path}')",
            "has_home = 'HOME' in env\nprint(f'has_home={has_home}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "path=/usr/bin" in nb_runner.get_output(1)
        assert "has_home=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "has_home=True" in nb_runner.get_output(2)

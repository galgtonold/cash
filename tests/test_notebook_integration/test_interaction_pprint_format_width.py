"""
Interaction test: pprint formatting with width/depth control.
Tests pprint.pformat with various width, depth, compact settings,
and cross-cell pretty-printing of complex nested structures.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestPprintFormatWidth:
    """Test pprint formatting across cells."""

    def test_pprint_width(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: create nested structure
            "import pprint\ndata = {'alpha': [1, 2, 3], 'beta': [4, 5, 6], 'gamma': [7, 8, 9]}\nwide = pprint.pformat(data, width=120)\nnarrow = pprint.pformat(data, width=30)\nwide_lines = len(wide.splitlines())\nnarrow_lines = len(narrow.splitlines())\nprint(f'wide_lines={wide_lines}')\nprint(f'narrow_more={narrow_lines > wide_lines}')",
            # Cell 2: depth control
            "nested = {'a': {'b': {'c': {'d': 1}}}}\nshallow = pprint.pformat(nested, depth=2)\nhas_ellipsis = '...' in shallow\nprint(f'has_ellipsis={has_ellipsis}')",
            # Cell 3: compact mode
            "nums = list(range(15))\ncompact_str = pprint.pformat(nums, width=40, compact=True)\nnormal_str = pprint.pformat(nums, width=40, compact=False)\ncompact_lines = len(compact_str.splitlines())\nnormal_lines = len(normal_str.splitlines())\nprint(f'compact_fewer={compact_lines <= normal_lines}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "narrow_more=True" in out1
        out2 = nb_runner.get_output(2)
        assert "has_ellipsis=True" in out2
        out3 = nb_runner.get_output(3)
        assert "compact_fewer=True" in out3

    def test_pprint_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import pprint\nitems = {'x': 10, 'y': 20, 'z': 30}\nformatted = pprint.pformat(items, width=60)\nprint(f'has_x={\"x\" in formatted}')\nprint(f'has_z={\"z\" in formatted}')",
            "char_count = len(formatted)\nprint(f'chars={char_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1a = nb_runner.get_output(1)
        assert "has_x=True" in out1a

        # Add more items
        nb_runner.set_cell_source(1, "import pprint\nitems = {'x': 10, 'y': 20, 'z': 30, 'w': 40, 'v': 50}\nformatted = pprint.pformat(items, width=60)\nprint(f'has_x={\"x\" in formatted}')\nprint(f'has_v={\"v\" in formatted}')")
        nb_runner.run_cells([1, 2])
        out1b = nb_runner.get_output(1)
        assert "has_v=True" in out1b

    def test_pprint_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import pprint\nobj = [{'key': i, 'val': i * 10} for i in range(5)]\npretty = pprint.pformat(obj, width=50)\nline_count = len(pretty.splitlines())\nprint(f'line_count={line_count}')",
            "has_key_3 = 'key' in pretty and '3' in pretty\nprint(f'has_key_3={has_key_3}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "line_count=" in out1
        out2 = nb_runner.get_output(2)
        assert "has_key_3=True" in out2

        # Re-run - cache
        nb_runner.run_all()
        assert "has_key_3=True" in nb_runner.get_output(2)

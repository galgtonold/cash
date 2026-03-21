"""
Interaction test: string center, ljust, rjust padding operations.
Tests fixed-width string formatting with different fill characters,
cross-cell alignment pipelines, and cache invalidation on width changes.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringCenterLjustRjust:
    """Test string alignment methods across cells."""

    def test_alignment_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: basic alignment
            "text = 'Hi'\ncentered = text.center(10, '*')\nleft = text.ljust(10, '-')\nright = text.rjust(10, '.')\nprint(f'centered={centered}')\nprint(f'left={left}')\nprint(f'right={right}')",
            # Cell 2: use aligned strings
            "c_len = len(centered)\nl_len = len(left)\nr_len = len(right)\nall_same = c_len == l_len == r_len\nprint(f'all_len_10={all_same}')",
            # Cell 3: build table row
            "name = 'Item'.ljust(10)\nprice = '$9.99'.rjust(10)\nrow = name + '|' + price\nprint(f'row={row}')\nprint(f'row_len={len(row)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "centered=****Hi****" in out1
        assert "left=Hi--------" in out1
        assert "right=........Hi" in out1
        out2 = nb_runner.get_output(2)
        assert "all_len_10=True" in out2
        out3 = nb_runner.get_output(3)
        assert "row_len=21" in out3

    def test_alignment_edit(self, nb_runner):
        nb_runner.create_notebook([
            "word = 'OK'\naligned = word.center(8, '=')\nprint(f'aligned={aligned}')",
            "stripped = aligned.strip('=')\nprint(f'stripped={stripped}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "aligned====OK===" in nb_runner.get_output(1)
        assert "stripped=OK" in nb_runner.get_output(2)

        # Edit to use rjust
        nb_runner.set_cell_source(1, "word = 'OK'\naligned = word.rjust(8, '=')\nprint(f'aligned={aligned}')")
        nb_runner.run_cells([1, 2])
        assert "aligned=======OK" in nb_runner.get_output(1)
        assert "stripped=OK" in nb_runner.get_output(2)

    def test_alignment_cache(self, nb_runner):
        nb_runner.create_notebook([
            "label = 'Test'\npadded = label.center(12)\nprint(f'padded_len={len(padded)}')",
            "is_centered = padded.strip() == 'Test'\nprint(f'is_centered={is_centered}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "padded_len=12" in nb_runner.get_output(1)
        assert "is_centered=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_centered=True" in nb_runner.get_output(2)

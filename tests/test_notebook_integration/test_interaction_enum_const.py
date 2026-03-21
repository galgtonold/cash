"""Batch 233 – Enum definition and constant map edit tests.

Tests editing enum definitions, constant maps, and named values
used in downstream calculations.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestEnumConstEdits:
    """Editing enums and constant mappings."""

    def test_edit_enum_add_member(self, nb_runner):
        """Edit an enum to add a new member."""
        nb_runner.create_notebook([
            "from enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3",
            "colors = [c.name for c in Color]\nprint(f'colors = {colors}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "RED" in nb_runner.get_output(2)
        assert "BLUE" in nb_runner.get_output(2)

        # Add YELLOW
        nb_runner.set_cell_source(1, "from enum import Enum\nclass Color(Enum):\n    RED = 1\n    GREEN = 2\n    BLUE = 3\n    YELLOW = 4")
        nb_runner.run_all()
        assert "YELLOW" in nb_runner.get_output(2)

    def test_edit_status_code_lookup(self, nb_runner):
        """Edit a constant mapping lookup."""
        nb_runner.create_notebook([
            "STATUS_CODES = {200: 'OK', 404: 'Not Found', 500: 'Server Error'}",
            "msg = STATUS_CODES.get(200, 'Unknown')\nprint(f'msg = {msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "msg = OK" in nb_runner.get_output(2)

        # Change lookup key
        nb_runner.set_cell_source(2, "msg = STATUS_CODES.get(404, 'Unknown')\nprint(f'msg = {msg}')")
        nb_runner.run_all()
        assert "msg = Not Found" in nb_runner.get_output(2)

    def test_edit_tax_rate_constant(self, nb_runner):
        """Edit a named constant used in calculations."""
        nb_runner.create_notebook([
            "TAX_RATE = 0.08",
            "price = 100\ntax = price * TAX_RATE\ntotal = price + tax\nprint(f'total = {total:.2f}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 108.00" in nb_runner.get_output(2)

        # Change tax rate
        nb_runner.set_cell_source(1, "TAX_RATE = 0.10")
        nb_runner.run_all()
        assert "total = 110.00" in nb_runner.get_output(2)

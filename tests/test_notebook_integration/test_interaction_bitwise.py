"""Batch 202 – Bit manipulation and bitwise operation interaction tests.

Tests editing bitwise operations (AND, OR, XOR, shifts)
and their propagation.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestBitwiseEdits:
    """Editing bitwise operations."""

    def test_edit_bitwise_op(self, nb_runner):
        """Edit the bitwise operator."""
        nb_runner.create_notebook([
            "a = 0b1100\nb = 0b1010  # bitwise source",
            "result = a & b\nprint(f'result = {bin(result)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 0b1000" in nb_runner.get_output(2)

        # Change to OR
        nb_runner.set_cell_source(
            2, "result = a | b\nprint(f'result = {bin(result)}')"
        )
        nb_runner.run_all()
        assert "result = 0b1110" in nb_runner.get_output(2)

    def test_edit_shift_amount(self, nb_runner):
        """Edit shift amounts."""
        nb_runner.create_notebook([
            "val = 1  # shift source",
            "result = val << 3\nprint(f'result = {result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 8" in nb_runner.get_output(2)

        # Change shift amount
        nb_runner.set_cell_source(
            2, "result = val << 10\nprint(f'result = {result}')"
        )
        nb_runner.run_all()
        assert "result = 1024" in nb_runner.get_output(2)

    def test_edit_xor_mask(self, nb_runner):
        """Edit XOR mask."""
        nb_runner.create_notebook([
            "data = 0xFF  # xor source",
            "mask = 0x0F  # xor mask",
            "result = data ^ mask\nprint(f'result = {hex(result)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 0xf0" in nb_runner.get_output(3)

        # Change mask
        nb_runner.set_cell_source(2, "mask = 0xF0  # xor mask v2")
        nb_runner.run_all()
        assert "result = 0xf" in nb_runner.get_output(3)

    def test_edit_bitwise_chain(self, nb_runner):
        """Edit a chain of bitwise operations."""
        nb_runner.create_notebook([
            "flags = 0b0000  # bitwise chain source",
            "flags = flags | 0b0001  # set bit 0",
            "flags = flags | 0b0100  # set bit 2",
            "print(f'flags = {bin(flags)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flags = 0b101" in nb_runner.get_output(4)

        # Change to set different bits
        nb_runner.set_cell_source(2, "flags = flags | 0b0010  # set bit 1 instead")
        nb_runner.run_all()
        assert "flags = 0b110" in nb_runner.get_output(4)

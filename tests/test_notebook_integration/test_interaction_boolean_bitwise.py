"""Batch 389: boolean and bitwise operations across cells."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestBooleanBitwise:
    def test_boolean_logic(self, nb_runner):
        nb_runner.create_notebook([
            "flags = {'admin': True, 'active': True, 'verified': False}",
            "can_edit = flags['admin'] and flags['active']\ncan_publish = flags['admin'] and flags['verified']\nprint(f'edit={can_edit} publish={can_publish}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "edit=True publish=False" in nb_runner.get_output(2)

    def test_bitwise_ops_edit(self, nb_runner):
        nb_runner.create_notebook([
            "a = 0b1100\nb = 0b1010",
            "and_op = a & b\nor_op = a | b\nxor_op = a ^ b\nprint(f'and={bin(and_op)} or={bin(or_op)} xor={bin(xor_op)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "and=0b1000" in out
        assert "or=0b1110" in out
        assert "xor=0b110" in out
        # Edit
        nb_runner.set_cell_source(1, "a = 0b1111\nb = 0b0101")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "and=0b101" in out2
        assert "or=0b1111" in out2

    def test_bit_shift(self, nb_runner):
        nb_runner.create_notebook([
            "val = 8",
            "left = val << 2\nright = val >> 1\nprint(f'left={left} right={right}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "left=32 right=4" in nb_runner.get_output(2)

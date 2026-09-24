"""Boolean logic and bitwise operators across cells."""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


@pytest.mark.upstream
class TestBitwiseEdits:
    """Editing bitwise operations."""

    def test_edit_bitwise_op(self, nb_runner):
        """Edit the bitwise operator."""
        nb_runner.create_notebook(
            [
                "a = 0b1100\nb = 0b1010  # bitwise source",
                "result = a & b\nprint(f'result = {bin(result)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 0b1000" in nb_runner.get_output(2)

        # Change to OR
        nb_runner.set_cell_source(2, "result = a | b\nprint(f'result = {bin(result)}')")
        nb_runner.run_all()
        assert "result = 0b1110" in nb_runner.get_output(2)

    def test_edit_shift_amount(self, nb_runner):
        """Edit shift amounts."""
        nb_runner.create_notebook(
            [
                "val = 1  # shift source",
                "result = val << 3\nprint(f'result = {result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 8" in nb_runner.get_output(2)

        # Change shift amount
        nb_runner.set_cell_source(2, "result = val << 10\nprint(f'result = {result}')")
        nb_runner.run_all()
        assert "result = 1024" in nb_runner.get_output(2)

    def test_edit_xor_mask(self, nb_runner):
        """Edit XOR mask."""
        nb_runner.create_notebook(
            [
                "data = 0xFF  # xor source",
                "mask = 0x0F  # xor mask",
                "result = data ^ mask\nprint(f'result = {hex(result)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result = 0xf0" in nb_runner.get_output(3)

        # Change mask
        nb_runner.set_cell_source(2, "mask = 0xF0  # xor mask v2")
        nb_runner.run_all()
        assert "result = 0xf" in nb_runner.get_output(3)

    def test_edit_bitwise_chain(self, nb_runner):
        """Edit a chain of bitwise operations."""
        nb_runner.create_notebook(
            [
                "flags = 0b0000  # bitwise chain source",
                "flags = flags | 0b0001  # set bit 0",
                "flags = flags | 0b0100  # set bit 2",
                "print(f'flags = {bin(flags)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "flags = 0b101" in nb_runner.get_output(4)

        # Change to set different bits
        nb_runner.set_cell_source(2, "flags = flags | 0b0010  # set bit 1 instead")
        nb_runner.run_all()
        assert "flags = 0b110" in nb_runner.get_output(4)


class TestBooleanBitwise:
    """boolean and bitwise operations across cells."""

    def test_boolean_logic(self, nb_runner):
        nb_runner.create_notebook(
            [
                "flags = {'admin': True, 'active': True, 'verified': False}",
                "can_edit = flags['admin'] and flags['active']\ncan_publish = flags['admin'] and flags['verified']\nprint(f'edit={can_edit} publish={can_publish}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "edit=True publish=False" in nb_runner.get_output(2)

    def test_bitwise_ops_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "a = 0b1100\nb = 0b1010",
                "and_op = a & b\nor_op = a | b\nxor_op = a ^ b\nprint(f'and={bin(and_op)} or={bin(or_op)} xor={bin(xor_op)}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "val = 8",
                "left = val << 2\nright = val >> 1\nprint(f'left={left} right={right}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "left=32 right=4" in nb_runner.get_output(2)


class TestBooleanLogicEdits:
    """Boolean logic patterns with edit propagation."""

    def test_compound_condition_edit(self, nb_runner):
        """Edit compound boolean condition."""
        nb_runner.create_notebook(
            [
                "data = [15, 25, 35, 45, 55, 65, 75]",
                "lo = 20\nhi = 60",
                "in_range = [x for x in data if lo <= x <= hi]\nprint(f'in_range = {in_range}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "in_range = [25, 35, 45, 55]" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, "lo = 30\nhi = 70")
        nb_runner.run_all()
        assert "in_range = [35, 45, 55, 65]" in nb_runner.get_output(3)

    def test_any_all_edit(self, nb_runner):
        """Edit data, any/all results change."""
        nb_runner.create_notebook(
            [
                "scores = [80, 90, 70, 85, 95]",
                "all_pass = all(s >= 60 for s in scores)\nany_perfect = any(s == 100 for s in scores)\nprint(f'all_pass={all_pass} any_perfect={any_perfect}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "all_pass=True any_perfect=False" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "scores = [80, 90, 50, 85, 100]")
        nb_runner.run_all()
        assert "all_pass=False any_perfect=True" in nb_runner.get_output(2)

    def test_boolean_function_edit(self, nb_runner):
        """Edit function with boolean logic."""
        nb_runner.create_notebook(
            [
                "def classify(n):\n    if n > 0:\n        return 'positive'\n    elif n < 0:\n        return 'negative'\n    return 'zero'",
                "labels = [classify(x) for x in [-5, 0, 5]]\nprint(f'labels = {labels}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "labels = ['negative', 'zero', 'positive']" in nb_runner.get_output(2)

        nb_runner.set_cell_source(
            1,
            "def classify(n):\n    if n >= 0:\n        return 'non-negative'\n    return 'negative'",
        )
        nb_runner.run_all()
        assert "labels = ['negative', 'non-negative', 'non-negative']" in nb_runner.get_output(2)

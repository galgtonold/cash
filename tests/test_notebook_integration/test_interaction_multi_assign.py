"""Batch 187 – Multi-assignment & augmented assignment interaction tests.

Tests editing multi-target assignments, augmented assignments (+=, *=),
and walrus operator patterns.
"""

import pytest

pytestmark = [pytest.mark.stress, pytest.mark.upstream, pytest.mark.timeout(90)]


class TestMultiAssignmentEdits:
    """Editing multi-assignment patterns."""

    def test_edit_multi_assign(self, nb_runner):
        """Edit a multi-assignment statement."""
        nb_runner.create_notebook([
            "a = b = c = 10  # multi assign",
            "total = a + b + c\nprint(f'total = {total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total = 30" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "a = b = c = 20  # multi assign v2")
        nb_runner.run_all()
        assert "total = 60" in nb_runner.get_output(2)

    def test_edit_swap_assignment(self, nb_runner):
        """Edit a swap assignment."""
        nb_runner.create_notebook([
            "x, y = 1, 2  # swap source",
            "x, y = y, x\nprint(f'x={x} y={y}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=2 y=1" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "x, y = 10, 20  # swap source v2")
        nb_runner.run_all()
        assert "x=20 y=10" in nb_runner.get_output(2)


class TestAugmentedAssignmentEdits:
    """Editing augmented assignment operations."""

    def test_edit_augmented_op(self, nb_runner):
        """Edit the augmented assignment operator."""
        nb_runner.create_notebook([
            "val = 10  # augmented source",
            "val += 5\nprint(f'val = {val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val = 15" in nb_runner.get_output(2)

        nb_runner.set_cell_source(2, "val *= 5\nprint(f'val = {val}')")
        nb_runner.run_all()
        assert "val = 50" in nb_runner.get_output(2)

    def test_edit_augmented_source(self, nb_runner):
        """Edit the source value for augmented assignment."""
        nb_runner.create_notebook([
            "base = 100  # augmented base",
            "base //= 3\nprint(f'base = {base}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "base = 33" in nb_runner.get_output(2)

        nb_runner.set_cell_source(1, "base = 200  # augmented base v2")
        nb_runner.run_all()
        assert "base = 66" in nb_runner.get_output(2)

    def test_chain_augmented_assignments(self, nb_runner):
        """Chain of augmented assignments across cells."""
        nb_runner.create_notebook([
            "n = 1  # chain augmented start",
            "n += 9  # step 1",
            "n *= 2  # step 2",
            "print(f'n = {n}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        # 1+9=10, 10*2=20
        assert "n = 20" in nb_runner.get_output(4)

        # Edit middle step
        nb_runner.set_cell_source(2, "n += 99  # step 1 v2")
        nb_runner.run_all()
        # 1+99=100, 100*2=200
        assert "n = 200" in nb_runner.get_output(4)

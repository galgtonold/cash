"""
Batch 281: Copy/deepcopy interaction tests.
Tests that cache invalidation works correctly when objects are copied
and the original is modified vs when the copy is modified.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestCopyDeepcopyInteraction:
    """Test copy/deepcopy patterns with cache invalidation."""

    def test_shallow_copy_edit_original(self, nb_runner):
        """Editing original after shallow copy should invalidate original-dependent cells."""
        nb_runner.create_notebook([
            "import copy\noriginal = [1, 2, 3]",
            "shallow = copy.copy(original)",
            "result = sum(original) + sum(shallow)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=12" in out

        # Edit original
        nb_runner.set_cell_source(1, "import copy\noriginal = [10, 20, 30]")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "result=120" in out

    def test_deepcopy_edit_original(self, nb_runner):
        """Deepcopy should create independent object; editing original propagates."""
        nb_runner.create_notebook([
            "import copy\ndata = {'a': [1, 2], 'b': [3, 4]}",
            "clone = copy.deepcopy(data)",
            "total = sum(data['a']) + sum(clone['b'])",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=10" in out

        # Edit data
        nb_runner.set_cell_source(1, "import copy\ndata = {'a': [10, 20], 'b': [30, 40]}")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "total=100" in out

    def test_copy_chain_propagation(self, nb_runner):
        """Chain: original -> copy1 -> copy2; edit original propagates through."""
        nb_runner.create_notebook([
            "import copy\nsrc = [1, 2, 3]",
            "c1 = copy.copy(src)",
            "c2 = copy.copy(c1)",
            "val = sum(c2)",
            "print(f'val={val}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "val=6" in out

        nb_runner.set_cell_source(1, "import copy\nsrc = [10, 20, 30]")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "val=60" in out

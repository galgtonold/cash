"""Batch 433: tuple operations immutability and named access."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestTupleOpsImmutable:
    def test_tuple_count_index(self, nb_runner):
        nb_runner.create_notebook([
            "t = (1, 2, 3, 2, 1, 2)",
            "c = t.count(2)\ni = t.index(3)\nprint(f'count={c} index={i}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=3" in nb_runner.get_output(2)
        assert "index=2" in nb_runner.get_output(2)

    def test_tuple_concat_repeat(self, nb_runner):
        nb_runner.create_notebook([
            "a = (1, 2)\nb = (3, 4)",
            "combined = a + b\nrepeated = a * 3\nprint(f'combined={combined} repeated={repeated}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "combined=(1, 2, 3, 4)" in nb_runner.get_output(2)
        assert "repeated=(1, 2, 1, 2, 1, 2)" in nb_runner.get_output(2)

    def test_tuple_edit(self, nb_runner):
        nb_runner.create_notebook([
            "coords = (10, 20, 30)",
            "x, y, z = coords\ntotal = x + y + z\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=60" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "coords = (100, 200, 300)")
        nb_runner.run_all()
        assert "total=600" in nb_runner.get_output(2)

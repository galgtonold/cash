"""Batch 438: dict merge operators and update patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDictMergeUpdate:
    def test_merge_operator(self, nb_runner):
        nb_runner.create_notebook([
            "d1 = {'a': 1, 'b': 2}\nd2 = {'b': 20, 'c': 30}",
            "merged = d1 | d2\nprint(f'merged={merged}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "'a': 1" in nb_runner.get_output(2)
        assert "'b': 20" in nb_runner.get_output(2)
        assert "'c': 30" in nb_runner.get_output(2)

    def test_dict_comprehension_merge(self, nb_runner):
        nb_runner.create_notebook([
            "keys = ['x', 'y', 'z']\nvals = [10, 20, 30]",
            "d = {k: v for k, v in zip(keys, vals)}\nreversed_d = {v: k for k, v in d.items()}\nprint(f'd={d} rev={reversed_d}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "'x': 10" in out
        assert "10: 'x'" in out

    def test_dict_merge_edit(self, nb_runner):
        nb_runner.create_notebook([
            "base = {'mode': 'fast', 'verbose': True}",
            "override = {'verbose': False}\nfinal = base | override\nprint(f'mode={final[\"mode\"]} verbose={final[\"verbose\"]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "mode=fast" in nb_runner.get_output(2)
        assert "verbose=False" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "base = {'mode': 'slow', 'verbose': True}")
        nb_runner.run_all()
        assert "mode=slow" in nb_runner.get_output(2)

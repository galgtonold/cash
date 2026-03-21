"""Batch 394: unpacking generalization (**kwargs, *args) across cells."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestUnpackGeneralization:
    def test_kwargs_merge(self, nb_runner):
        nb_runner.create_notebook([
            "defaults = {'color': 'red', 'size': 10}\noverrides = {'size': 20, 'weight': 5}",
            "merged = {**defaults, **overrides}\nprint(f'merged={dict(sorted(merged.items()))}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "merged={'color': 'red', 'size': 20, 'weight': 5}" in nb_runner.get_output(2)

    def test_args_spread_edit(self, nb_runner):
        nb_runner.create_notebook([
            "parts = ([1, 2], [3, 4], [5])",
            "combined = [*parts[0], *parts[1], *parts[2]]\nprint(f'combined={combined}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "combined=[1, 2, 3, 4, 5]" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "parts = ([10], [20, 30])")
        nb_runner.set_cell_source(2, "combined = [*parts[0], *parts[1]]\nprint(f'combined={combined}')")
        nb_runner.run_all()
        assert "combined=[10, 20, 30]" in nb_runner.get_output(2)

    def test_func_kwargs(self, nb_runner):
        nb_runner.create_notebook([
            "def build_url(scheme='http', host='localhost', port=80):\n    return f'{scheme}://{host}:{port}'",
            "params = {'scheme': 'https', 'host': 'example.com', 'port': 443}\nurl = build_url(**params)\nprint(f'url={url}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "url=https://example.com:443" in nb_runner.get_output(2)

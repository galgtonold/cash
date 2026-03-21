"""Batch 446: string removeprefix and removesuffix (3.9+)."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringRemovePrefixSuffix:
    def test_removeprefix(self, nb_runner):
        nb_runner.create_notebook([
            "text = 'TestCaseExample'",
            "no_prefix = text.removeprefix('Test')\nno_miss = text.removeprefix('Foo')\nprint(f'no_prefix={no_prefix} no_miss={no_miss}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "no_prefix=CaseExample" in nb_runner.get_output(2)
        assert "no_miss=TestCaseExample" in nb_runner.get_output(2)

    def test_removesuffix(self, nb_runner):
        nb_runner.create_notebook([
            "filename = 'data.csv'",
            "no_ext = filename.removesuffix('.csv')\nno_miss = filename.removesuffix('.txt')\nprint(f'no_ext={no_ext} no_miss={no_miss}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "no_ext=data" in nb_runner.get_output(2)
        assert "no_miss=data.csv" in nb_runner.get_output(2)

    def test_remove_edit(self, nb_runner):
        nb_runner.create_notebook([
            "s = 'prefix_content_suffix'",
            "r = s.removeprefix('prefix_').removesuffix('_suffix')\nprint(f'r={r}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "r=content" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "s = 'prefix_hello_world_suffix'")
        nb_runner.run_all()
        assert "r=hello_world" in nb_runner.get_output(2)

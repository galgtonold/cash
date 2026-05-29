"""Batch 419: string maketrans and translate."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringMaketransTranslate:


    def test_translate_edit(self, nb_runner):
        nb_runner.create_notebook([
            "msg = 'abc'",
            "table = str.maketrans('abc', 'xyz')\nresult = msg.translate(table)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=xyz" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "msg = 'aabbcc'")
        nb_runner.run_all()
        assert "result=xxyyzz" in nb_runner.get_output(2)

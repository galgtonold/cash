"""Batch 342: string methods chain and text processing edits."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringMethodChain:
    def test_string_chain_basic(self, nb_runner):
        nb_runner.create_notebook([
            "text = '  Hello, World!  '",
            "processed = text.strip().lower().replace(',', '').replace('!', '')\nwords = processed.split()\nprint(f'words={words}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "words=['hello', 'world']" in nb_runner.get_output(2)

    def test_string_chain_edit(self, nb_runner):
        nb_runner.create_notebook([
            "text = 'foo-bar-baz'",
            "parts = text.split('-')\nresult = '_'.join(p.upper() for p in parts)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=FOO_BAR_BAZ" in nb_runner.get_output(2)
        # Edit input
        nb_runner.set_cell_source(1, "text = 'alpha-beta'")
        nb_runner.run_all()
        assert "result=ALPHA_BETA" in nb_runner.get_output(2)

    def test_multiline_string_processing(self, nb_runner):
        nb_runner.create_notebook([
            "lines = 'line1\\nline2\\nline3\\nline2\\nline1'",
            "unique = list(dict.fromkeys(lines.split('\\n')))\ncount = len(unique)\nprint(f'unique={unique} count={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "unique=['line1', 'line2', 'line3']" in out
        assert "count=3" in out

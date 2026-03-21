"""Batch 363: string partition, join patterns, and format_map."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringPartitionJoin:
    def test_partition(self, nb_runner):
        nb_runner.create_notebook([
            "text = 'key=value=extra'",
            "key, sep, rest = text.partition('=')\nprint(f'key={key} rest={rest}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "key=key rest=value=extra" in nb_runner.get_output(2)

    def test_join_edit(self, nb_runner):
        nb_runner.create_notebook([
            "words = ['hello', 'beautiful', 'world']",
            "sentence = ' '.join(words)\ncsv_line = ','.join(words)\nprint(f'sentence={sentence}')\nprint(f'csv={csv_line}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "sentence=hello beautiful world" in nb_runner.get_output(2)
        assert "csv=hello,beautiful,world" in nb_runner.get_output(2)
        # Edit words
        nb_runner.set_cell_source(1, "words = ['foo', 'bar']")
        nb_runner.run_all()
        assert "sentence=foo bar" in nb_runner.get_output(2)
        assert "csv=foo,bar" in nb_runner.get_output(2)

    def test_format_map_pattern(self, nb_runner):
        nb_runner.create_notebook([
            "template = '{name} scored {score} points'\ndata = {'name': 'Alice', 'score': 95}",
            "result = template.format_map(data)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=Alice scored 95 points" in nb_runner.get_output(2)

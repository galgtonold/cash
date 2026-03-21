"""Batch 371: multi-step data pipeline with intermediate transforms."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestDataPipelineSteps:
    def test_pipeline_3_steps(self, nb_runner):
        nb_runner.create_notebook([
            "raw = [' Alice:85 ', ' Bob:92 ', ' Charlie:78 ']",
            "cleaned = [s.strip() for s in raw]",
            "parsed = [{'name': s.split(':')[0], 'score': int(s.split(':')[1])} for s in cleaned]",
            "avg = sum(p['score'] for p in parsed) / len(parsed)\nprint(f'avg={round(avg, 1)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avg=85.0" in nb_runner.get_output(4)

    def test_pipeline_edit_input(self, nb_runner):
        nb_runner.create_notebook([
            "data = [1, -2, 3, -4, 5]",
            "positives = [x for x in data if x > 0]",
            "doubled = [x * 2 for x in positives]\nprint(f'doubled={doubled}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "doubled=[2, 6, 10]" in nb_runner.get_output(3)
        # Edit
        nb_runner.set_cell_source(1, "data = [-10, 20, -30, 40]")
        nb_runner.run_all()
        assert "doubled=[40, 80]" in nb_runner.get_output(3)

    def test_pipeline_edit_middle(self, nb_runner):
        nb_runner.create_notebook([
            "numbers = [10, 20, 30, 40, 50]",
            "transformed = [x + 5 for x in numbers]",
            "total = sum(transformed)\nprint(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total=175" in nb_runner.get_output(3)
        # Edit middle step
        nb_runner.set_cell_source(2, "transformed = [x * 2 for x in numbers]")
        nb_runner.run_all()
        assert "total=300" in nb_runner.get_output(3)

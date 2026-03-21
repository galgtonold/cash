"""
Batch 312: String chain operations interaction tests.
Tests split→join, replace→strip chains with cache invalidation.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringChainOpsInteraction:
    """Test string chain operations with cache invalidation."""

    def test_split_upper_join_edit(self, nb_runner):
        """Editing input string should propagate through split/upper/join."""
        nb_runner.create_notebook([
            "raw = 'hello world python coding'",
            "words = raw.split()",
            "upper_words = [w.upper() for w in words]",
            "result = '-'.join(upper_words)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=HELLO-WORLD-PYTHON-CODING" in out

        nb_runner.set_cell_source(1, "raw = 'foo bar baz'")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=FOO-BAR-BAZ" in out

    def test_replace_chain_edit(self, nb_runner):
        """Editing replace targets should propagate."""
        nb_runner.create_notebook([
            "text = 'Hello, World! Hello, Python!'",
            "step1 = text.replace('Hello', 'Hi')",
            "step2 = step1.replace('!', '.')",
            "print(f'out={step2}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "out=Hi, World. Hi, Python." in out

        nb_runner.set_cell_source(2, "step1 = text.replace('Hello', 'Greetings')")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "out=Greetings, World. Greetings, Python." in out

    def test_strip_format_edit(self, nb_runner):
        """Editing whitespace handling should propagate."""
        nb_runner.create_notebook([
            "raw_lines = ['  Alice  ', '  Bob  ', '  Charlie  ']",
            "cleaned = [line.strip() for line in raw_lines]",
            "formatted = [f'[{name}]' for name in cleaned]",
            "result = ', '.join(formatted)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=[Alice], [Bob], [Charlie]" in out

        nb_runner.set_cell_source(1, "raw_lines = ['  X  ', '  Y  ']")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=[X], [Y]" in out

"""Batch 479: string maketrans and translate cipher."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringMaketransCipher:
    def test_caesar_cipher(self, nb_runner):
        nb_runner.create_notebook([
            "import string",
            "shift = 3\nalpha = string.ascii_lowercase\nshifted = alpha[shift:] + alpha[:shift]\ntable = str.maketrans(alpha, shifted)\nencoded = 'hello world'.translate(table)\nprint(f'encoded={encoded}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "encoded=khoor zruog" in nb_runner.get_output(2)

    def test_remove_chars(self, nb_runner):
        nb_runner.create_notebook([
            "text = 'Hello, World! 123'",
            "table = str.maketrans('', '', '!,. ')\ncleaned = text.translate(table)\nprint(f'cleaned={cleaned}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cleaned=HelloWorld123" in nb_runner.get_output(2)

    def test_cipher_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import string",
            "shift = 1\nalpha = string.ascii_lowercase\nshifted = alpha[shift:] + alpha[:shift]\ntable = str.maketrans(alpha, shifted)\nresult = 'abc'.translate(table)\nprint(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=bcd" in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "shift = 2\nalpha = string.ascii_lowercase\nshifted = alpha[shift:] + alpha[:shift]\ntable = str.maketrans(alpha, shifted)\nresult = 'abc'.translate(table)\nprint(f'result={result}')")
        nb_runner.run_all()
        assert "result=cde" in nb_runner.get_output(2)

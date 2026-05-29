"""Batch 386: string translate and maketrans patterns."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringTranslate:

    def test_translate_edit(self, nb_runner):
        nb_runner.create_notebook([
            "cipher_shift = 3",
            "import string\nlower = string.ascii_lowercase\nshifted = lower[cipher_shift:] + lower[:cipher_shift]\ntable = str.maketrans(lower, shifted)\nencoded = 'hello'.translate(table)\nprint(f'encoded={encoded}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "encoded=khoor" in nb_runner.get_output(2)
        # Edit shift
        nb_runner.set_cell_source(1, "cipher_shift = 1")
        nb_runner.run_all()
        assert "encoded=ifmmp" in nb_runner.get_output(2)

    def test_remove_punctuation(self, nb_runner):
        nb_runner.create_notebook([
            "import string\ntext = 'Hello, World! How are you?'",
            "no_punct = text.translate(str.maketrans('', '', string.punctuation))\nwords = no_punct.split()\nprint(f'words={words}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "words=['Hello', 'World', 'How', 'are', 'you']" in nb_runner.get_output(2)

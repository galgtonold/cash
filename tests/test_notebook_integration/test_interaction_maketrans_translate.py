"""
Interaction test: string maketrans with translate and multi-char replacement.
Tests str.maketrans with 3-arg form (intab, outtab, delchars), translate,
and cross-cell text transformation pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringMaketransTranslate:
    """Test str.maketrans and translate across cells."""

    def test_maketrans_translate_pipeline(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: create translation table
            "table = str.maketrans('aeiou', '12345', '!?.')\nprint(f'table_size={len(table)}')",
            # Cell 2: apply translations
            "text = 'Hello, World! Are you okay?'\ntranslated = text.translate(table)\nprint(f'result={translated}')",
            # Cell 3: analyze
            "vowel_count = sum(1 for c in 'Hello, World! Are you okay?' if c in 'aeiou')\ndigit_count = sum(1 for c in translated if c.isdigit())\npunct_removed = all(c not in translated for c in '!?.')\nprint(f'vowels={vowel_count}')\nprint(f'digits={digit_count}')\nprint(f'punct_clean={punct_removed}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "table_size=" in out1
        out2 = nb_runner.get_output(2)
        assert "result=" in out2
        out3 = nb_runner.get_output(3)
        assert "punct_clean=True" in out3

    def test_maketrans_edit_mapping(self, nb_runner):
        nb_runner.create_notebook([
            "table = str.maketrans({'a': '@', 'e': '3', 'i': '!', 'o': '0', 's': '$'})\nprint(f'table_type={type(table).__name__}')",
            "text = 'secret message'\nencoded = text.translate(table)\nprint(f'encoded={encoded}')",
            "diff_count = sum(1 for a, b in zip(text, encoded) if a != b)\nprint(f'changes={diff_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert "encoded=$3cr3t m3$$@g3" in out2

        # Edit mapping
        nb_runner.set_cell_source(1, "table = str.maketrans({'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5'})\nprint(f'table_type={type(table).__name__}')")
        nb_runner.run_cells([1, 2, 3])
        assert "encoded=53cr3t m3554g3" in nb_runner.get_output(2)

    def test_maketrans_cache(self, nb_runner):
        nb_runner.create_notebook([
            "rot13_in = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'\nrot13_out = 'nopqrstuvwxyzabcdefghijklmNOPQRSTUVWXYZABCDEFGHIJKLM'\ntable = str.maketrans(rot13_in, rot13_out)\nprint('rot13_ready')",
            "msg = 'Hello World'\nencoded = msg.translate(table)\ndecoded = encoded.translate(table)\nprint(f'encoded={encoded}')\nprint(f'roundtrip={decoded == msg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "encoded=Uryyb Jbeyq" in out
        assert "roundtrip=True" in out

        # Re-run - cache
        nb_runner.run_all()
        assert "roundtrip=True" in nb_runner.get_output(2)

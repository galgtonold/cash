"""
Interaction test: re.sub with function replacement.
Tests re.sub with callable replacement, groups, backreferences,
and cross-cell regex transformation pipelines.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestReSubFunctionReplace:
    """Test re.sub with function replacement across cells."""

    def test_re_sub_function(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: sub with function replacement
            "import re\ndef double_num(m):\n    return str(int(m.group()) * 2)\n\ntext = 'item1 costs 5 dollars and item2 costs 10 dollars'\nresult = re.sub(r'\\d+', double_num, text)\nprint(f'result={result}')",
            # Cell 2: sub with backreference
            "swapped = re.sub(r'(\\w+) costs (\\d+)', r'\\2 for \\1', text)\nprint(f'swapped={swapped}')",
            # Cell 3: count substitutions
            "count = 0\ndef counter_replace(m):\n    global count\n    count += 1\n    return f'[{count}]'\n\ncounted = re.sub(r'\\d+', counter_replace, text)\nprint(f'counted={counted}')\nprint(f'replacements={count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "item2" in out1 and "10" in out1 and "20" in out1
        out2 = nb_runner.get_output(2)
        assert "5 for item1" in out2
        out3 = nb_runner.get_output(3)
        assert "replacements=" in out3

    def test_re_sub_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import re\npattern = r'[aeiou]'\ntext = 'hello world'\nresult = re.sub(pattern, '*', text)\nprint(f'result={result}')",
            "vowel_free_len = len(result.replace('*', ''))\nprint(f'consonants={vowel_free_len}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "result=h*ll* w*rld" in nb_runner.get_output(1)
        # h,l,l,' ',w,r,l,d = 8 non-vowel chars
        assert "consonants=8" in nb_runner.get_output(2)

        # Edit pattern to uppercase too
        nb_runner.set_cell_source(1, "import re\npattern = r'[aeiouAEIOU]'\ntext = 'Hello World'\nresult = re.sub(pattern, '*', text)\nprint(f'result={result}')")
        nb_runner.run_cells([1, 2])
        assert "result=H*ll* W*rld" in nb_runner.get_output(1)

    def test_re_sub_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import re\nraw = '  hello   world  '\ncleaned = re.sub(r'\\s+', ' ', raw).strip()\nprint(f'cleaned={cleaned}')",
            "word_count = len(cleaned.split())\nprint(f'words={word_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "cleaned=hello world" in nb_runner.get_output(1)
        assert "words=2" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "words=2" in nb_runner.get_output(2)

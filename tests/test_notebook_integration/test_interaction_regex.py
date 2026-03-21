"""
Batch 284: Regex compilation interaction tests.
Tests that editing regex patterns properly invalidates match results downstream.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestRegexInteraction:
    """Test regex compilation and matching with cache invalidation."""

    def test_regex_pattern_edit(self, nb_runner):
        """Editing a regex pattern should invalidate match results."""
        nb_runner.create_notebook([
            "import re\npattern = re.compile(r'\\d+')",
            "text = 'abc 123 def 456'",
            "matches = pattern.findall(text)",
            "result = ','.join(matches)",
            "print(f'result={result}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=123,456" in out

        # Change pattern to match words instead
        nb_runner.set_cell_source(1, "import re\npattern = re.compile(r'[a-z]+')")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "result=abc,def" in out

    def test_regex_sub_edit(self, nb_runner):
        """Editing substitution pattern should propagate."""
        nb_runner.create_notebook([
            "import re\nreplacer = re.compile(r'\\s+')",
            "text = 'hello   world   python'",
            "cleaned = replacer.sub('-', text)",
            "print(f'cleaned={cleaned}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "cleaned=hello-world-python" in out

        # Change to replace with underscore
        nb_runner.set_cell_source(1, "import re\nreplacer = re.compile(r'\\s+')")
        nb_runner.set_cell_source(3, "cleaned = replacer.sub('_', text)")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "cleaned=hello_world_python" in out

    def test_regex_groups_edit(self, nb_runner):
        """Editing regex with groups should propagate captured groups."""
        nb_runner.create_notebook([
            "import re\npat = re.compile(r'(\\w+)@(\\w+\\.\\w+)')",
            "email = 'alice@example.com'",
            "m = pat.match(email)\nuser = m.group(1)\ndomain = m.group(2)",
            "print(f'user={user},domain={domain}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "user=alice,domain=example.com" in out

        nb_runner.set_cell_source(2, "email = 'bob@work.org'")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "user=bob,domain=work.org" in out

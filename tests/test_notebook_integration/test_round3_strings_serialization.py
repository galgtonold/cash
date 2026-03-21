"""
Batch 29: String formatting, regex, serialization, and I/O patterns
across notebook cells.
"""
import pytest
import textwrap

pytestmark = [pytest.mark.integration, pytest.mark.stress]


class TestStringFormattingPatterns:
    """Test string formatting propagation across cells."""

    def test_fstring_with_complex_expressions(self, nb_runner):
        """f-strings with complex expressions across cells."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                data = {'name': 'Alice', 'scores': [90, 85, 92]}
            """),
            textwrap.dedent("""\
                avg = sum(data['scores']) / len(data['scores'])
                report = f"{data['name']}: avg={avg:.1f}, total={sum(data['scores'])}"
                print(report)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Alice: avg=89.0, total=267" in nb_runner.get_output(2)

    def test_template_string_pattern(self, nb_runner):
        """String.Template across cells."""
        nb_runner.create_notebook([
            "from string import Template",
            textwrap.dedent("""\
                tmpl = Template("Hello $name, you have $count items")
            """),
            textwrap.dedent("""\
                msg = tmpl.substitute(name='Bob', count=5)
                print(msg)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Hello Bob, you have 5 items" in nb_runner.get_output(3)

    def test_multiline_string_operations(self, nb_runner):
        """Multi-line string operations across cells."""
        nb_runner.create_notebook([
            textwrap.dedent('''\
                text = """
                Line one
                Line two
                Line three
                """
            '''),
            textwrap.dedent("""\
                lines = [l.strip() for l in text.strip().split('\\n') if l.strip()]
                print(f"count={len(lines)} first={lines[0]}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=3 first=Line one" in nb_runner.get_output(2)


class TestRegexPatterns:
    """Test regex patterns across cells."""

    def test_compiled_regex_across_cells(self, nb_runner):
        """Compiled regex used in subsequent cell."""
        nb_runner.create_notebook([
            "import re",
            "pattern = re.compile(r'(\\d{4})-(\\d{2})-(\\d{2})')",
            textwrap.dedent("""\
                text = "Born on 1990-05-15, graduated 2012-06-01"
                dates = pattern.findall(text)
                print(dates)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "1990" in output
        assert "2012" in output

    def test_regex_change_propagation(self, nb_runner):
        """Change regex pattern → downstream updates."""
        nb_runner.create_notebook([
            "import re",
            "pattern = re.compile(r'\\b[A-Z][a-z]+\\b')",
            textwrap.dedent("""\
                text = "Hello World foo Bar"
                matches = pattern.findall(text)
                print(matches)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "Hello" in output
        assert "World" in output
        assert "Bar" in output

        # Change to only match 5+ char capitalized words
        nb_runner.set_cell_source(2, "pattern = re.compile(r'\\b[A-Z][a-z]{4,}\\b')")
        nb_runner.run_all()
        output2 = nb_runner.get_output(3)
        assert "Hello" in output2
        assert "World" in output2
        # "Bar" is only 3 chars, should NOT match
        assert "Bar" not in output2


class TestJsonPatterns:
    """Test JSON serialization across cells."""

    def test_json_roundtrip(self, nb_runner, tmp_path):
        """JSON write + read across cells with file tracking."""
        json_path = tmp_path / "data.json"
        path_str = str(json_path).replace('\\', '/')

        nb_runner.create_notebook([
            "import json",
            textwrap.dedent(f"""\
                data = {{'name': 'test', 'values': [1, 2, 3]}}
                with open('{path_str}', 'w') as f:
                    json.dump(data, f)
            """),
            textwrap.dedent(f"""\
                with open('{path_str}') as f:
                    loaded = json.load(f)
                print(loaded)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "'name': 'test'" in output or '"name": "test"' in output

    def test_json_transform_pipeline(self, nb_runner):
        """JSON transform across multiple cells."""
        nb_runner.create_notebook([
            "import json",
            textwrap.dedent("""\
                raw = '[{"name": "a", "val": 1}, {"name": "b", "val": 2}]'
                records = json.loads(raw)
            """),
            textwrap.dedent("""\
                transformed = {r['name']: r['val'] * 10 for r in records}
                print(transformed)
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "'a': 10" in output
        assert "'b': 20" in output


class TestCsvPatterns:
    """Test CSV reading/writing without pandas."""

    def test_csv_stdlib_across_cells(self, nb_runner, tmp_path):
        """csv module read/write across cells."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("name,score\nAlice,90\nBob,85\n")
        path_str = str(csv_path).replace('\\', '/')

        nb_runner.create_notebook([
            "import csv",
            textwrap.dedent(f"""\
                with open('{path_str}') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
            """),
            textwrap.dedent("""\
                avg = sum(int(r['score']) for r in rows) / len(rows)
                print(f"avg={avg:.1f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avg=87.5" in nb_runner.get_output(3)


class TestPicklePatterns:
    """Test pickle serialization across cells."""

    def test_pickle_roundtrip(self, nb_runner, tmp_path):
        """Pickle object and reload it."""
        pkl_path = tmp_path / "obj.pkl"
        path_str = str(pkl_path).replace('\\', '/')

        nb_runner.create_notebook([
            "import pickle",
            textwrap.dedent(f"""\
                data = {{'key': [1, 2, 3], 'nested': {{'x': 42}}}}
                with open('{path_str}', 'wb') as f:
                    pickle.dump(data, f)
            """),
            textwrap.dedent(f"""\
                with open('{path_str}', 'rb') as f:
                    loaded = pickle.load(f)
                print(loaded['nested']['x'])
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42" in nb_runner.get_output(3)


class TestPathLibPatterns:
    """Test pathlib usage across cells."""

    def test_pathlib_operations(self, nb_runner, tmp_path):
        """pathlib.Path operations across cells."""
        path_str = str(tmp_path).replace('\\', '/')

        nb_runner.create_notebook([
            "from pathlib import Path",
            f"base = Path('{path_str}')",
            textwrap.dedent("""\
                # Create some files
                (base / 'a.txt').write_text('hello')
                (base / 'b.txt').write_text('world')
                (base / 'c.py').write_text('# code')
            """),
            textwrap.dedent("""\
                txt_files = sorted(base.glob('*.txt'))
                print(f"txt_count={len(txt_files)}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "txt_count=2" in nb_runner.get_output(4)

    def test_pathlib_change_propagation(self, nb_runner, tmp_path):
        """Change base path → downstream glob updates."""
        sub1 = tmp_path / "sub1"
        sub2 = tmp_path / "sub2"
        sub1.mkdir()
        sub2.mkdir()
        (sub1 / "f1.txt").write_text("a")
        (sub2 / "f2.txt").write_text("b")
        (sub2 / "f3.txt").write_text("c")

        path1 = str(sub1).replace('\\', '/')
        path2 = str(sub2).replace('\\', '/')

        nb_runner.create_notebook([
            "from pathlib import Path",
            f"base = Path('{path1}')",
            textwrap.dedent("""\
                count = len(list(base.glob('*.txt')))
                print(f"count={count}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=1" in nb_runner.get_output(3)

        nb_runner.set_cell_source(2, f"base = Path('{path2}')")
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(3)

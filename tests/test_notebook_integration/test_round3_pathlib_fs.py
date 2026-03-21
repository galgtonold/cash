"""Batch 94 – pathlib and file system operations."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration, pytest.mark.files]


class TestPathlibPatterns:
    """pathlib usage patterns."""

    def test_pathlib_operations(self, nb_runner, tmp_path):
        """Path construction, existence checks, iteration."""
        work = tmp_path / "fs_test"
        work.mkdir()
        work_str = str(work).replace('\\', '/')
        nb_runner.create_notebook([
            textwrap.dedent(f"""\
                from pathlib import Path
                base = Path('{work_str}')
                # Create directory structure
                (base / 'src').mkdir(exist_ok=True)
                (base / 'src' / 'main.py').write_text('print("hello")')
                (base / 'src' / 'utils.py').write_text('x = 1')
                (base / 'data').mkdir(exist_ok=True)
                (base / 'data' / 'input.txt').write_text('data here')
                (base / 'README.md').write_text('# Project')
            """),
            textwrap.dedent(f"""\
                from pathlib import Path
                base = Path('{work_str}')
                all_files = sorted([p.name for p in base.rglob('*') if p.is_file()])
                py_files = sorted([p.name for p in base.glob('**/*.py')])
                readme_exists = (base / 'README.md').exists()
            """),
            "print(f'all={all_files}')\nprint(f'py={py_files}')\nprint(f'readme={readme_exists}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "main.py" in out
        assert "utils.py" in out
        assert "input.txt" in out
        assert "readme=True" in out

    def test_pathlib_stem_suffix(self, nb_runner, tmp_path):
        """Path parts: stem, suffix, parent."""
        work = tmp_path / "path_parts"
        work.mkdir()
        work_str = str(work).replace('\\', '/')
        nb_runner.create_notebook([
            textwrap.dedent(f"""\
                from pathlib import Path
                p = Path('{work_str}') / 'archive' / 'data_2024.csv.gz'
                stem = p.stem
                suffix = p.suffix
                suffixes = p.suffixes
                parent_name = p.parent.name
                parts_count = len(p.parts)
            """),
            "print(f'stem={stem} suffix={suffix} suffixes={suffixes} parent={parent_name}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "stem=data_2024.csv" in out
        assert "suffix=.gz" in out
        assert ".csv" in out
        assert "parent=archive" in out


class TestFileSystemOps:
    """File system read/write with caching."""

    def test_file_write_read_chain(self, nb_runner, tmp_path):
        """Write files, read back, process."""
        work = tmp_path / "rw_test"
        work.mkdir()
        work_str = str(work).replace('\\', '/')
        nb_runner.create_notebook([
            textwrap.dedent(f"""\
                from pathlib import Path
                base = Path('{work_str}')
                for i in range(5):
                    (base / f'log_{{i}}.txt').write_text(f'Entry {{i}}: value={{i*10}}')
            """),
            textwrap.dedent(f"""\
                from pathlib import Path
                base = Path('{work_str}')
                contents = []
                for f in sorted(base.glob('log_*.txt')):
                    contents.append(f.read_text())
                total_files = len(contents)
            """),
            "print(f'files={total_files}')\nprint(f'first={contents[0]}')\nprint(f'last={contents[-1]}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "files=5" in out
        assert "Entry 0" in out
        assert "Entry 4" in out

    def test_temp_file_processing(self, nb_runner, tmp_path):
        """Process data through temp files."""
        work = tmp_path / "temp_proc"
        work.mkdir()
        work_str = str(work).replace('\\', '/')
        nb_runner.create_notebook([
            textwrap.dedent(f"""\
                from pathlib import Path
                import json
                data = {{'items': [1, 2, 3, 4, 5], 'meta': 'test'}}
                path = Path('{work_str}') / 'data.json'
                path.write_text(json.dumps(data))
            """),
            textwrap.dedent(f"""\
                from pathlib import Path
                import json
                path = Path('{work_str}') / 'data.json'
                loaded = json.loads(path.read_text())
                total = sum(loaded['items'])
                meta = loaded['meta']
            """),
            "print(f'total={total} meta={meta}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "total=15" in out
        assert "meta=test" in out

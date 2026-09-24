"""Files reached through pathlib and shelve."""

import textwrap

import pytest


# pathlib and file system operations.
class TestPathlibPatterns:
    """pathlib usage patterns."""

    @pytest.mark.stress
    @pytest.mark.integration
    @pytest.mark.files
    def test_pathlib_operations(self, nb_runner, tmp_path):
        """Path construction, existence checks, iteration."""
        work = tmp_path / "fs_test"
        work.mkdir()
        work_str = str(work).replace("\\", "/")
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "main.py" in out
        assert "utils.py" in out
        assert "input.txt" in out
        assert "readme=True" in out

    @pytest.mark.stress
    @pytest.mark.integration
    @pytest.mark.files
    def test_pathlib_stem_suffix(self, nb_runner, tmp_path):
        """Path parts: stem, suffix, parent."""
        work = tmp_path / "path_parts"
        work.mkdir()
        work_str = str(work).replace("\\", "/")
        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "stem=data_2024.csv" in out
        assert "suffix=.gz" in out
        assert ".csv" in out
        assert "parent=archive" in out

    # Pathlib & IO patterns — cash caching with pathlib, io, tempfile.
    @pytest.mark.stress
    def test_pathlib_basic_operations(self, nb_runner, tmp_path):
        """Pathlib path construction and operations."""
        dir_path = str(tmp_path).replace("\\", "/")
        nb_runner.create_notebook(
            [
                "from pathlib import Path",
                textwrap.dedent(f"""\
                base = Path('{dir_path}')
                sub = base / 'data' / 'output'
                sub.mkdir(parents=True, exist_ok=True)
                exists = sub.exists()
                print(f"exists={{exists}} name={{sub.name}}")
            """),
                textwrap.dedent("""\
                # Create files
                for i in range(3):
                    (sub / f'file_{i}.txt').write_text(f'content_{i}')
                files = sorted(f.name for f in sub.iterdir())
                print(f"files={files}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "exists=True name=output" in nb_runner.get_output(2)
        assert "files=['file_0.txt', 'file_1.txt', 'file_2.txt']" in nb_runner.get_output(3)

    @pytest.mark.stress
    def test_pathlib_glob(self, nb_runner, tmp_path):
        """Pathlib glob pattern matching."""
        glob_dir = tmp_path / "glob_test"
        glob_dir.mkdir()
        dir_path = str(glob_dir).replace("\\", "/")
        nb_runner.create_notebook(
            [
                "from pathlib import Path",
                textwrap.dedent(f"""\
                base = Path('{dir_path}')
                # Create mixed files
                (base / 'data.csv').write_text('a,b\\n1,2')
                (base / 'report.csv').write_text('x,y\\n3,4')
                (base / 'notes.txt').write_text('hello')
                (base / 'config.json').write_text('{{}}')
            """),
                textwrap.dedent("""\
                csv_files = sorted(f.name for f in base.glob('*.csv'))
                all_files = sorted(f.name for f in base.glob('*.*'))
                print(f"csv={csv_files}")
                print(f"total={len(all_files)}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "csv=['data.csv', 'report.csv']" in nb_runner.get_output(3)
        assert "total=4" in nb_runner.get_output(3)

    @pytest.mark.stress
    def test_pathlib_read_write(self, nb_runner, tmp_path):
        """Pathlib read_text/write_text across cells."""
        dir_path = str(tmp_path).replace("\\", "/")
        nb_runner.create_notebook(
            [
                "from pathlib import Path",
                textwrap.dedent(f"""\
                p = Path('{dir_path}') / 'data.txt'
                lines = ['line1', 'line2', 'line3']
                p.write_text('\\n'.join(lines))
                size = p.stat().st_size
                print(f"size={{size}}")
            """),
                textwrap.dedent("""\
                content = p.read_text()
                line_count = len(content.strip().split('\\n'))
                print(f"lines={line_count} first={content.split(chr(10))[0]}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "size=" in nb_runner.get_output(2)
        assert "lines=3 first=line1" in nb_runner.get_output(3)


# Interaction test: shelve module for persistent dict-like storage.
# Tests shelve.open with writeback, cross-cell key access,
# and cache invalidation when shelf contents change.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestShelvePersistentDict:
    """Test shelve persistent storage across cells."""

    def test_shelve_ops(self, nb_runner, tmp_path):
        shelf_path = str(tmp_path / "test_shelf").replace("\\", "/")
        nb_runner.create_notebook(
            [
                # Cell 1: write to shelf
                f"import shelve\nshelf_path = '{shelf_path}'\nwith shelve.open(shelf_path) as db:\n    db['name'] = 'Alice'\n    db['scores'] = [90, 85, 95]\n    key_count = len(db)\nprint(f'keys={{key_count}}')",
                # Cell 2: read from shelf
                "with shelve.open(shelf_path) as db:\n    name = db['name']\n    scores = db['scores']\nprint(f'name={name}')\nprint(f'avg={sum(scores)/len(scores):.1f}')",
                # Cell 3: check keys
                "with shelve.open(shelf_path) as db:\n    all_keys = sorted(db.keys())\nprint(f'all_keys={all_keys}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "keys=2" in out1
        out2 = nb_runner.get_output(2)
        assert "name=Alice" in out2
        assert "avg=90.0" in out2
        out3 = nb_runner.get_output(3)
        assert "name" in out3
        assert "scores" in out3

    def test_shelve_edit(self, nb_runner, tmp_path):
        shelf_path = str(tmp_path / "edit_shelf").replace("\\", "/")
        nb_runner.create_notebook(
            [
                f"import shelve\nshelf_path = '{shelf_path}'\nwith shelve.open(shelf_path) as db:\n    db['val'] = 100\n    stored = db['val']\nprint(f'stored={{stored}}')",
                "doubled = stored * 2\nprint(f'doubled={doubled}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "stored=100" in nb_runner.get_output(1)
        assert "doubled=200" in nb_runner.get_output(2)

        # Edit stored value
        nb_runner.set_cell_source(
            1,
            f"import shelve\nshelf_path = '{shelf_path}'\nwith shelve.open(shelf_path) as db:\n    db['val'] = 250\n    stored = db['val']\nprint(f'stored={{stored}}')",
        )
        nb_runner.run_cells([1, 2])
        assert "stored=250" in nb_runner.get_output(1)
        assert "doubled=500" in nb_runner.get_output(2)

    def test_shelve_cache(self, nb_runner, tmp_path):
        shelf_path = str(tmp_path / "cache_shelf").replace("\\", "/")
        nb_runner.create_notebook(
            [
                f"import shelve\nwith shelve.open('{shelf_path}') as db:\n    db['x'] = 42\n    x = db['x']\nprint(f'x={{x}}')",
                "is_42 = x == 42\nprint(f'is_42={is_42}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x=42" in nb_runner.get_output(1)
        assert "is_42=True" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "is_42=True" in nb_runner.get_output(2)

"""Batch 59: Pathlib & IO patterns — cash caching with pathlib, io, tempfile."""
import textwrap
import pytest


@pytest.mark.stress
class TestPathlibPatterns:
    """Test pathlib operations across cells."""

    def test_pathlib_basic_operations(self, nb_runner, tmp_path):
        """Pathlib path construction and operations."""
        dir_path = str(tmp_path).replace("\\", "/")
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "exists=True name=output" in nb_runner.get_output(2)
        assert "files=['file_0.txt', 'file_1.txt', 'file_2.txt']" in nb_runner.get_output(3)

    def test_pathlib_glob(self, nb_runner, tmp_path):
        """Pathlib glob pattern matching."""
        glob_dir = tmp_path / "glob_test"
        glob_dir.mkdir()
        dir_path = str(glob_dir).replace("\\", "/")
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "csv=['data.csv', 'report.csv']" in nb_runner.get_output(3)
        assert "total=4" in nb_runner.get_output(3)

    def test_pathlib_read_write(self, nb_runner, tmp_path):
        """Pathlib read_text/write_text across cells."""
        dir_path = str(tmp_path).replace("\\", "/")
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "size=" in nb_runner.get_output(2)
        assert "lines=3 first=line1" in nb_runner.get_output(3)


@pytest.mark.stress
class TestIOPatterns:
    """Test io module patterns."""

    def test_stringio_across_cells(self, nb_runner):
        """StringIO as in-memory file across cells."""
        nb_runner.create_notebook([
            "import io",
            textwrap.dedent("""\
                buf = io.StringIO()
                buf.write("Hello, ")
                buf.write("World!")
                buf.write("\\nLine 2")
                content = buf.getvalue()
                print(f"len={len(content)}")
            """),
            textwrap.dedent("""\
                lines = content.split('\\n')
                print(f"lines={lines}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "len=" in nb_runner.get_output(2)
        assert "Hello, World!" in nb_runner.get_output(3)

    def test_bytesio_across_cells(self, nb_runner):
        """BytesIO as in-memory binary file."""
        nb_runner.create_notebook([
            "import io",
            textwrap.dedent("""\
                buf = io.BytesIO()
                buf.write(b'\\x00\\x01\\x02\\x03')
                buf.write(b'\\xff\\xfe')
                byte_count = buf.tell()
                print(f"bytes={byte_count}")
            """),
            textwrap.dedent("""\
                buf.seek(0)
                data = buf.read()
                hex_str = data.hex()
                print(f"hex={hex_str}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "bytes=6" in nb_runner.get_output(2)
        assert "hex=000102" in nb_runner.get_output(3)

    def test_csv_writer_stringio(self, nb_runner):
        """CSV writer with StringIO across cells."""
        nb_runner.create_notebook([
            "import csv\nimport io",
            textwrap.dedent("""\
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow(['name', 'age', 'city'])
                writer.writerow(['Alice', 30, 'NYC'])
                writer.writerow(['Bob', 25, 'LA'])
                csv_output = buf.getvalue()
                print(f"lines={csv_output.count(chr(10))}")
            """),
            textwrap.dedent("""\
                buf2 = io.StringIO(csv_output)
                reader = csv.DictReader(buf2)
                rows = list(reader)
                names = [r['name'] for r in rows]
                print(f"names={names}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names=['Alice', 'Bob']" in nb_runner.get_output(3)

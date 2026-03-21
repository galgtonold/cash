"""
Interaction test: io.StringIO and io.BytesIO in-memory streams.
Tests reading/writing to in-memory buffers, seek/tell operations,
cross-cell stream sharing, and cache invalidation on content changes.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestIoStringBytesIO:
    """Test io module in-memory streams across cells."""

    def test_stringio_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: write to StringIO
            "import io\nbuf = io.StringIO()\nbuf.write('Hello ')\nbuf.write('World')\ncontent = buf.getvalue()\nprint(f'content={content}')\nprint(f'length={len(content)}')",
            # Cell 2: read from StringIO
            "reader = io.StringIO(content)\nfirst_word = reader.read(5)\nprint(f'first={first_word}')\nrest = reader.read()\nprint(f'rest={rest}')",
            # Cell 3: BytesIO
            "bbuf = io.BytesIO(b'binary data')\nbbuf.seek(7)\nchunk = bbuf.read()\nprint(f'chunk={chunk}')\nprint(f'chunk_str={chunk.decode()}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "content=Hello World" in out1
        assert "length=11" in out1
        out2 = nb_runner.get_output(2)
        assert "first=Hello" in out2
        assert "rest= World" in out2
        out3 = nb_runner.get_output(3)
        assert "chunk_str=data" in out3

    def test_stringio_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import io\nbuf = io.StringIO()\nfor i in range(3):\n    buf.write(f'line{i}\\n')\ntext = buf.getvalue()\nline_count = text.strip().count('\\n') + 1\nprint(f'lines={line_count}')",
            "first_line = text.split('\\n')[0]\nprint(f'first={first_line}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "lines=3" in nb_runner.get_output(1)
        assert "first=line0" in nb_runner.get_output(2)

        # Edit range
        nb_runner.set_cell_source(1, "import io\nbuf = io.StringIO()\nfor i in range(5):\n    buf.write(f'line{i}\\n')\ntext = buf.getvalue()\nline_count = text.strip().count('\\n') + 1\nprint(f'lines={line_count}')")
        nb_runner.run_cells([1, 2])
        assert "lines=5" in nb_runner.get_output(1)
        assert "first=line0" in nb_runner.get_output(2)

    def test_stringio_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import io\ncsv_data = 'a,b,c\\n1,2,3\\n4,5,6'\nreader = io.StringIO(csv_data)\nheader = reader.readline().strip()\nprint(f'header={header}')",
            "cols = header.split(',')\ncol_count = len(cols)\nprint(f'col_count={col_count}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "header=a,b,c" in nb_runner.get_output(1)
        assert "col_count=3" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "col_count=3" in nb_runner.get_output(2)

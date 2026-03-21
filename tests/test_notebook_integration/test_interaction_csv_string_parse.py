"""Batch 378: csv-like string parsing without file I/O."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestCsvStringParsing:
    def test_csv_parse(self, nb_runner):
        nb_runner.create_notebook([
            "import csv\nimport io\ncsv_text = 'name,age,city\\nAlice,30,NY\\nBob,25,LA'",
            "reader = csv.DictReader(io.StringIO(csv_text))\nrows = list(reader)\nnames = [r['name'] for r in rows]\nprint(f'names={names}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names=['Alice', 'Bob']" in nb_runner.get_output(2)

    def test_csv_edit_data(self, nb_runner):
        nb_runner.create_notebook([
            "import csv\nimport io\ncsv_text = 'a,b\\n1,2\\n3,4'",
            "reader = csv.reader(io.StringIO(csv_text))\nheader = next(reader)\ndata = [list(map(int, row)) for row in reader]\ntotal = sum(sum(row) for row in data)\nprint(f'header={header} total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "header=['a', 'b']" in nb_runner.get_output(2)
        assert "total=10" in nb_runner.get_output(2)
        # Edit
        nb_runner.set_cell_source(1, "import csv\nimport io\ncsv_text = 'x,y\\n10,20\\n30,40\\n50,60'")
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "header=['x', 'y']" in out
        assert "total=210" in out

    def test_csv_write_string(self, nb_runner):
        nb_runner.create_notebook([
            "import csv\nimport io\nrows = [['name', 'val'], ['a', '1'], ['b', '2']]",
            "buf = io.StringIO()\nwriter = csv.writer(buf)\nwriter.writerows(rows)\nresult = buf.getvalue().strip()\nprint(f'result={repr(result)}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name,val" in nb_runner.get_output(2)

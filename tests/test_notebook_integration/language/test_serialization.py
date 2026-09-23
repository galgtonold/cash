"""json, csv text and pickle across cells."""

import textwrap

import pytest


# String formatting, regex, serialization, and I/O patterns
# across notebook cells.
@pytest.mark.integration
@pytest.mark.stress
class TestStringFormattingPatterns:
    """Test string formatting propagation across cells."""

    def test_fstring_with_complex_expressions(self, nb_runner):
        """f-strings with complex expressions across cells."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                data = {'name': 'Alice', 'scores': [90, 85, 92]}
            """),
                textwrap.dedent("""\
                avg = sum(data['scores']) / len(data['scores'])
                report = f"{data['name']}: avg={avg:.1f}, total={sum(data['scores'])}"
                print(report)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "Alice: avg=89.0, total=267" in nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.stress
class TestRegexPatterns:
    """Test regex patterns across cells."""

    def test_regex_change_propagation(self, nb_runner):
        """Change regex pattern → downstream updates."""
        nb_runner.create_notebook(
            [
                "import re",
                "pattern = re.compile(r'\\b[A-Z][a-z]+\\b')",
                textwrap.dedent("""\
                text = "Hello World foo Bar"
                matches = pattern.findall(text)
                print(matches)
            """),
            ]
        )
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


@pytest.mark.integration
@pytest.mark.stress
class TestJsonPatterns:
    """Test JSON serialization across cells."""

    def test_json_roundtrip(self, nb_runner, tmp_path):
        """JSON write + read across cells with file tracking."""
        json_path = tmp_path / "data.json"
        path_str = str(json_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(3)
        assert "'name': 'test'" in output or '"name": "test"' in output


@pytest.mark.integration
@pytest.mark.stress
class TestCsvPatterns:
    """Test CSV reading/writing without pandas."""

    def test_csv_stdlib_across_cells(self, nb_runner, tmp_path):
        """csv module read/write across cells."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("name,score\nAlice,90\nBob,85\n", encoding="utf-8")
        path_str = str(csv_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "avg=87.5" in nb_runner.get_output(3)


# Pickle/serialization edge cases — cash caching with pickle, struct, json.
class TestPicklePatterns:
    """Test pickle serialization across cells."""

    @pytest.mark.integration
    @pytest.mark.stress
    def test_pickle_roundtrip(self, nb_runner, tmp_path):
        """Pickle object and reload it."""
        pkl_path = tmp_path / "obj.pkl"
        path_str = str(pkl_path).replace("\\", "/")

        nb_runner.create_notebook(
            [
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
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "42" in nb_runner.get_output(3)

    @pytest.mark.stress
    def test_pickle_roundtrip_reports_save_and_load(self, nb_runner, tmp_path):
        """Pickle dump and load across cells."""
        pkl_path = str(tmp_path / "data.pkl").replace("\\", "/")
        nb_runner.create_notebook(
            [
                "import pickle",
                textwrap.dedent(f"""\
                data = {{'name': 'test', 'values': [1, 2, 3], 'nested': {{'a': 10}}}}
                with open('{pkl_path}', 'wb') as f:
                    pickle.dump(data, f)
                print("saved")
            """),
                textwrap.dedent(f"""\
                with open('{pkl_path}', 'rb') as f:
                    loaded = pickle.load(f)
                print(f"loaded={{loaded}}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "saved" in nb_runner.get_output(2)
        assert "loaded=" in nb_runner.get_output(3)
        assert "'name': 'test'" in nb_runner.get_output(3)

    @pytest.mark.stress
    def test_pickle_custom_class(self, nb_runner, tmp_path):
        """Pickle custom class instances."""
        pkl_path = str(tmp_path / "obj.pkl").replace("\\", "/")
        nb_runner.create_notebook(
            [
                "import pickle",
                textwrap.dedent(f"""\
                class Config:
                    def __init__(self, **kwargs):
                        self.__dict__.update(kwargs)
                    def __repr__(self):
                        items = ', '.join(f'{{k}}={{v}}' for k, v in sorted(self.__dict__.items()) if not k.startswith('_cash'))
                        return f"Config({{items}})"

                cfg = Config(lr=0.01, epochs=100, batch_size=32)
                with open('{pkl_path}', 'wb') as f:
                    pickle.dump(cfg, f)
                print(f"cfg={{cfg}}")
            """),
                textwrap.dedent(f"""\
                with open('{pkl_path}', 'rb') as f:
                    loaded_cfg = pickle.load(f)
                print(f"loaded={{loaded_cfg}}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(2)
        assert "lr=0.01" in out1
        out2 = nb_runner.get_output(3)
        assert "lr=0.01" in out2

    @pytest.mark.stress
    def test_pickle_bytes_transfer(self, nb_runner):
        """Pickle to bytes and back across cells."""
        nb_runner.create_notebook(
            [
                "import pickle",
                textwrap.dedent("""\
                original = {'key': [1, 2, 3], 'flag': True}
                pickled_bytes = pickle.dumps(original)
                byte_count = len(pickled_bytes)
                print(f"bytes={byte_count}")
            """),
                textwrap.dedent("""\
                restored = pickle.loads(pickled_bytes)
                print(f"match={restored == original} keys={sorted(restored.keys())}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "bytes=" in nb_runner.get_output(2)
        assert "match=True" in nb_runner.get_output(3)


@pytest.mark.stress
class TestJsonSerialization:
    """Test JSON serialization patterns."""

    def test_json_nested_serialization(self, nb_runner, tmp_path):
        """JSON file write/read with nested data."""
        json_path = str(tmp_path / "data.json").replace("\\", "/")
        nb_runner.create_notebook(
            [
                "import json",
                textwrap.dedent(f"""\
                config = {{
                    'database': {{'host': 'localhost', 'port': 5432}},
                    'features': ['auth', 'cache', 'logging'],
                    'limits': {{'max_conn': 100, 'timeout': 30}}
                }}
                with open('{json_path}', 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"keys={{sorted(config.keys())}}")
            """),
                textwrap.dedent(f"""\
                with open('{json_path}') as f:
                    loaded = json.load(f)
                print(f"host={{loaded['database']['host']}} features={{len(loaded['features'])}}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['database', 'features', 'limits']" in nb_runner.get_output(2)
        assert "host=localhost features=3" in nb_runner.get_output(3)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestCsvStringParsing:
    """csv-like string parsing without file I/O."""

    def test_csv_parse(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import csv\nimport io\ncsv_text = 'name,age,city\\nAlice,30,NY\\nBob,25,LA'",
                "reader = csv.DictReader(io.StringIO(csv_text))\nrows = list(reader)\nnames = [r['name'] for r in rows]\nprint(f'names={names}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "names=['Alice', 'Bob']" in nb_runner.get_output(2)

    def test_csv_edit_data(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import csv\nimport io\ncsv_text = 'a,b\\n1,2\\n3,4'",
                "reader = csv.reader(io.StringIO(csv_text))\nheader = next(reader)\ndata = [list(map(int, row)) for row in reader]\ntotal = sum(sum(row) for row in data)\nprint(f'header={header} total={total}')",
            ]
        )
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
        nb_runner.create_notebook(
            [
                "import csv\nimport io\nrows = [['name', 'val'], ['a', '1'], ['b', '2']]",
                "buf = io.StringIO()\nwriter = csv.writer(buf)\nwriter.writerows(rows)\nresult = buf.getvalue().strip()\nprint(f'result={repr(result)}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name,val" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestJsonCustomEncoders:
    """json serialization with custom encoders."""

    def test_json_dumps_loads(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import json\ndata = {'name': 'Alice', 'scores': [95, 87, 92]}",
                's = json.dumps(data, sort_keys=True)\nloaded = json.loads(s)\nprint(f\'name={loaded["name"]} scores={loaded["scores"]}\')',
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name=Alice" in nb_runner.get_output(2)
        assert "scores=[95, 87, 92]" in nb_runner.get_output(2)

    def test_json_indent(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import json\nobj = {'a': 1, 'b': [2, 3]}",
                "pretty = json.dumps(obj, indent=2)\nline_count = len(pretty.split('\\n'))\nprint(f'lines={line_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        count = int(nb_runner.get_output(2).split("lines=")[1].strip())
        assert count >= 5

    def test_json_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import json\nrecord = {'x': 10, 'y': 20}",
                "s = json.dumps(record)\nsize = len(s)\nprint(f'size={size}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        first_size = int(nb_runner.get_output(2).split("size=")[1].strip())
        nb_runner.set_cell_source(1, "import json\nrecord = {'x': 10, 'y': 20, 'z': 30, 'w': 40}")
        nb_runner.run_all()
        second_size = int(nb_runner.get_output(2).split("size=")[1].strip())
        assert second_size > first_size


# Interaction test: json loads and dumps with custom encoding.
# Tests json serialization with custom defaults, indent,
# sort_keys, and cross-cell data transformation pipelines.
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestJsonCustomEncoding:
    """Test json loads/dumps with custom encoding across cells."""

    def test_json_custom(self, nb_runner):
        nb_runner.create_notebook(
            [
                # Cell 1: custom json serialization
                "import json\nfrom datetime import datetime, date\n\nclass DateEncoder(json.JSONEncoder):\n    def default(self, obj):\n        if isinstance(obj, (datetime, date)):\n            return obj.isoformat()\n        return super().default(obj)\n\ndata = {'name': 'Alice', 'created': date(2024, 1, 15), 'scores': [95, 87, 92]}\njson_str = json.dumps(data, cls=DateEncoder, sort_keys=True)\nprint(f'json={json_str}')",
                # Cell 2: parse back
                "parsed = json.loads(json_str)\nprint(f'name={parsed[\"name\"]}')\nprint(f'created={parsed[\"created\"]}')\nprint(f'scores_sum={sum(parsed[\"scores\"])}')",
                # Cell 3: pretty print
                "pretty = json.dumps(parsed, indent=2, sort_keys=True)\nline_count = len(pretty.split('\\n'))\nprint(f'lines={line_count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "2024-01-15" in out1
        assert "Alice" in out1
        out2 = nb_runner.get_output(2)
        assert "name=Alice" in out2
        assert "scores_sum=274" in out2
        out3 = nb_runner.get_output(3)
        assert "lines=" in out3

    def test_json_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import json\nconfig = {'debug': True, 'port': 8080}\njson_str = json.dumps(config, sort_keys=True)\nprint(f'json={json_str}')",
                "parsed = json.loads(json_str)\nport = parsed['port']\nprint(f'port={port}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "port=8080" in nb_runner.get_output(2)

        # Edit config
        nb_runner.set_cell_source(
            1,
            "import json\nconfig = {'debug': False, 'port': 9090, 'host': 'localhost'}\njson_str = json.dumps(config, sort_keys=True)\nprint(f'json={json_str}')",
        )
        nb_runner.run_cells([1, 2])
        assert "port=9090" in nb_runner.get_output(2)

    def test_json_cache(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import json\nitems = [{'id': 1, 'val': 'a'}, {'id': 2, 'val': 'b'}]\njson_str = json.dumps(items)\nprint(f'length={len(json_str)}')",
                "back = json.loads(json_str)\ncount = len(back)\nprint(f'count={count}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "count=2" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestJsonCustomSerde:
    """json serialization/deserialization with custom objects."""

    def test_json_custom_encoder(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import json\nfrom datetime import date\nclass DateEncoder(json.JSONEncoder):\n    def default(self, obj):\n        if isinstance(obj, date):\n            return obj.isoformat()\n        return super().default(obj)",
                "data = {'name': 'event', 'date': date(2024, 1, 15)}\nresult = json.dumps(data, cls=DateEncoder)\nprint(f'result={result}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert '"date": "2024-01-15"' in nb_runner.get_output(2)

    def test_json_edit_data(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import json\ndata = {'a': 1, 'b': [2, 3]}",
                "encoded = json.dumps(data, sort_keys=True)\ndecoded = json.loads(encoded)\nprint(f'encoded={encoded}')\nprint(f'match={data == decoded}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert 'encoded={"a": 1, "b": [2, 3]}' in out
        assert "match=True" in out
        # Edit data
        nb_runner.set_cell_source(1, "import json\ndata = {'x': [10, 20], 'y': 'hello'}")
        nb_runner.run_all()
        out2 = nb_runner.get_output(2)
        assert '"x": [10, 20]' in out2
        assert "match=True" in out2

    def test_json_nested(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import json\nnested = {'level1': {'level2': {'value': 42}}}",
                "s = json.dumps(nested)\nback = json.loads(s)\nval = back['level1']['level2']['value']\nprint(f'val={val}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "val=42" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.timeout(90)
class TestJsonDumpsLoadsCustom:
    """json dumps loads with custom encoder."""

    def test_json_roundtrip(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import json",
                "data = {'name': 'Alice', 'scores': [95, 87, 92], 'meta': {'grade': 'A'}}\nencoded = json.dumps(data, sort_keys=True)\ndecoded = json.loads(encoded)\nprint(f'encoded_type={type(encoded).__name__}')\nprint(f'name={decoded[\"name\"]} scores_sum={sum(decoded[\"scores\"])}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "encoded_type=str" in out
        assert "name=Alice" in out
        assert "scores_sum=274" in out

    def test_json_indent(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import json",
                "data = {'a': 1, 'b': [2, 3]}\npretty = json.dumps(data, indent=2)\nlines = pretty.strip().split('\\n')\nprint(f'lines={len(lines)} has_indent={\"  \" in pretty}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "has_indent=True" in out

    def test_json_edit(self, nb_runner):
        nb_runner.create_notebook(
            [
                "import json",
                "d = {'x': 1}\ns = json.dumps(d)\nprint(f's={s}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert 's={"x": 1}' in nb_runner.get_output(2)
        nb_runner.set_cell_source(2, "d = {'x': 1, 'y': 2}\ns = json.dumps(d)\nprint(f's={s}')")
        nb_runner.run_all()
        assert '"y": 2' in nb_runner.get_output(2)


# JSON serialization/deserialization interaction tests.
# Tests that editing data that gets serialized to JSON and then deserialized
# properly invalidates downstream cells.
@pytest.mark.integration
@pytest.mark.stress
@pytest.mark.timeout(90)
class TestJsonSerializationInteraction:
    """Test JSON serialization patterns with cache invalidation."""

    def test_json_roundtrip_edit(self, nb_runner):
        """Editing data before JSON roundtrip should propagate."""
        nb_runner.create_notebook(
            [
                "import json\ndata = {'name': 'Alice', 'score': 95}",
                "serialized = json.dumps(data)",
                "restored = json.loads(serialized)",
                'print(f\'name={restored["name"]},score={restored["score"]}\')',
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "name=Alice,score=95" in out

        nb_runner.set_cell_source(1, "import json\ndata = {'name': 'Bob', 'score': 88}")
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "name=Bob,score=88" in out

    def test_json_nested_edit(self, nb_runner):
        """Editing nested JSON structure should propagate through deserialization."""
        nb_runner.create_notebook(
            [
                "import json\nconfig = {'db': {'host': 'localhost', 'port': 5432}, 'debug': True}",
                "text = json.dumps(config, indent=2)",
                "parsed = json.loads(text)",
                "info = f\"{parsed['db']['host']}:{parsed['db']['port']}\"",
                "print(f'info={info}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "info=localhost:5432" in out

        nb_runner.set_cell_source(
            1, "import json\nconfig = {'db': {'host': 'remote.io', 'port': 3306}, 'debug': False}"
        )
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "info=remote.io:3306" in out

    def test_json_list_of_dicts_edit(self, nb_runner):
        """Editing a list of dicts serialized as JSON."""
        nb_runner.create_notebook(
            [
                "import json\nrecords = [{'id': 1, 'val': 10}, {'id': 2, 'val': 20}]",
                "blob = json.dumps(records)",
                "loaded = json.loads(blob)",
                "total = sum(r['val'] for r in loaded)",
                "print(f'total={total}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "total=30" in out

        nb_runner.set_cell_source(
            1, "import json\nrecords = [{'id': 1, 'val': 100}, {'id': 2, 'val': 200}, {'id': 3, 'val': 300}]"
        )
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "total=600" in out


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(90)
class TestSerializationEdits:
    """Editing serialization round-trips."""

    def test_json_roundtrip_edit(self, nb_runner):
        """Edit data in a JSON serialization round-trip."""
        nb_runner.create_notebook(
            [
                "import json",
                "data = {'name': 'Alice', 'age': 30}",
                'serialized = json.dumps(data)\nrestored = json.loads(serialized)\nprint(f\'name={restored["name"]} age={restored["age"]}\')',
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "name=Alice age=30" in nb_runner.get_output(3)

        # Change data
        nb_runner.set_cell_source(2, "data = {'name': 'Bob', 'age': 25}")
        nb_runner.run_all()
        assert "name=Bob age=25" in nb_runner.get_output(3)

    def test_csv_roundtrip_edit(self, nb_runner, tmp_path):
        """Edit data in a CSV write/read round-trip."""
        fpath = str(tmp_path / "roundtrip.csv").replace("\\", "/")
        nb_runner.create_notebook(
            [
                f"import csv\nfpath = '{fpath}'",
                "with open(fpath, 'w', newline='') as f:\n    w = csv.writer(f)\n    w.writerow(['a', 'b'])\n    w.writerow([1, 2])",
                "with open(fpath) as f:\n    reader = csv.reader(f)\n    rows = list(reader)\nprint(f'rows = {rows}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "['a', 'b']" in out
        assert "['1', '2']" in out

        # Change written data
        nb_runner.set_cell_source(
            2,
            "with open(fpath, 'w', newline='') as f:\n    w = csv.writer(f)\n    w.writerow(['x', 'y', 'z'])\n    w.writerow([10, 20, 30])",
        )
        nb_runner.run_all()
        out2 = nb_runner.get_output(3)
        assert "['x', 'y', 'z']" in out2
        assert "['10', '20', '30']" in out2

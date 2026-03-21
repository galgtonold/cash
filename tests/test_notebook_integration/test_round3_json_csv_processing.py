"""Batch 91 – JSON/CSV data processing chains."""

import textwrap, pytest

pytestmark = [pytest.mark.stress, pytest.mark.integration]


class TestJsonProcessing:
    """JSON manipulation and processing patterns."""

    def test_json_roundtrip(self, nb_runner):
        """JSON serialize/deserialize roundtrip."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import json
                data = {
                    'users': [
                        {'name': 'Alice', 'age': 30, 'scores': [95, 87, 92]},
                        {'name': 'Bob', 'age': 25, 'scores': [78, 82, 90]},
                    ],
                    'metadata': {'version': '1.0', 'count': 2}
                }
                json_str = json.dumps(data, indent=2)
                restored = json.loads(json_str)
                match = data == restored
            """),
            "print(f'match={match}')\nprint(f'users={len(restored[\"users\"])}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "match=True" in out
        assert "users=2" in out

    def test_json_transform(self, nb_runner):
        """JSON transformation pipeline."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import json
                raw = '[{"name": "Alice", "score": 95}, {"name": "Bob", "score": 82}, {"name": "Charlie", "score": 78}]'
                records = json.loads(raw)
                # Transform: add grade based on score
                for r in records:
                    r['grade'] = 'A' if r['score'] >= 90 else 'B' if r['score'] >= 80 else 'C'
                grades = {r['name']: r['grade'] for r in records}
            """),
            "print(f'grades={grades}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "Alice" in out
        assert "'A'" in out
        assert "Charlie" in out

    def test_json_nested_query(self, nb_runner):
        """Query nested JSON structure."""
        nb_runner.create_notebook([
            textwrap.dedent("""\
                import json
                config = {
                    'database': {
                        'primary': {'host': 'db1.example.com', 'port': 5432},
                        'replica': {'host': 'db2.example.com', 'port': 5432},
                    },
                    'cache': {'host': 'redis.example.com', 'port': 6379},
                }

                def get_nested(d, path, default=None):
                    keys = path.split('.')
                    current = d
                    for k in keys:
                        if isinstance(current, dict) and k in current:
                            current = current[k]
                        else:
                            return default
                    return current

                primary_host = get_nested(config, 'database.primary.host')
                cache_port = get_nested(config, 'cache.port')
                missing = get_nested(config, 'database.tertiary.host', 'N/A')
            """),
            "print(f'primary={primary_host} cache_port={cache_port} missing={missing}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "primary=db1.example.com" in out
        assert "cache_port=6379" in out
        assert "missing=N/A" in out


class TestCsvProcessing:
    """CSV data processing patterns."""

    def test_csv_write_read(self, nb_runner, tmp_path):
        """Write CSV, read back, transform."""
        csv_path = str(tmp_path / "data" / "test.csv").replace('\\', '/')
        nb_runner.create_notebook([
            textwrap.dedent(f"""\
                import csv, os
                os.makedirs(os.path.dirname('{csv_path}'), exist_ok=True)
                rows = [
                    ['name', 'department', 'salary'],
                    ['Alice', 'Engineering', '95000'],
                    ['Bob', 'Marketing', '72000'],
                    ['Charlie', 'Engineering', '88000'],
                    ['Diana', 'Marketing', '78000'],
                ]
                with open('{csv_path}', 'w', newline='') as f:
                    csv.writer(f).writerows(rows)
            """),
            textwrap.dedent(f"""\
                import csv
                with open('{csv_path}', 'r') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
                eng_avg = sum(int(r['salary']) for r in data if r['department'] == 'Engineering') / sum(1 for r in data if r['department'] == 'Engineering')
            """),
            "print(f'count={len(data)} eng_avg={eng_avg}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(3)
        assert "count=4" in out
        assert "eng_avg=91500" in out

    def test_csv_propagation(self, nb_runner, tmp_path):
        """CSV with upstream filter change propagation."""
        csv_path = str(tmp_path / "data" / "scores.csv").replace('\\', '/')
        nb_runner.create_notebook([
            textwrap.dedent(f"""\
                import csv, os
                os.makedirs(os.path.dirname('{csv_path}'), exist_ok=True)
                with open('{csv_path}', 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['name', 'score'])
                    for name, score in [('A', 90), ('B', 75), ('C', 85), ('D', 60), ('E', 95)]:
                        w.writerow([name, score])
            """),
            "min_score = 80",
            textwrap.dedent(f"""\
                import csv
                with open('{csv_path}', 'r') as f:
                    data = list(csv.DictReader(f))
                passing = [r['name'] for r in data if int(r['score']) >= min_score]
            """),
            "print(f'passing={passing}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(4)
        assert "A" in out
        assert "C" in out
        assert "E" in out

        nb_runner.set_cell_source(2, "min_score = 90")
        nb_runner.run_cells([2, 3, 4])
        out2 = nb_runner.get_output(4)
        assert "A" in out2
        assert "E" in out2
        # B, C, D should no longer be in passing
        assert "B" not in out2
        assert "C" not in out2

"""Batch 46: Pickle/serialization edge cases — cash caching with pickle, struct, json."""
import textwrap
import pytest


@pytest.mark.stress
class TestPicklePatterns:
    """Test pickle serialization across cells."""

    def test_pickle_roundtrip(self, nb_runner, tmp_path):
        """Pickle dump and load across cells."""
        pkl_path = str(tmp_path / "data.pkl").replace("\\", "/")
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "saved" in nb_runner.get_output(2)
        assert "loaded=" in nb_runner.get_output(3)
        assert "'name': 'test'" in nb_runner.get_output(3)

    def test_pickle_custom_class(self, nb_runner, tmp_path):
        """Pickle custom class instances."""
        pkl_path = str(tmp_path / "obj.pkl").replace("\\", "/")
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(2)
        assert "lr=0.01" in out1
        out2 = nb_runner.get_output(3)
        assert "lr=0.01" in out2

    def test_pickle_bytes_transfer(self, nb_runner):
        """Pickle to bytes and back across cells."""
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "bytes=" in nb_runner.get_output(2)
        assert "match=True" in nb_runner.get_output(3)


@pytest.mark.stress
class TestStructPatterns:
    """Test struct packing/unpacking."""

    def test_struct_pack_unpack(self, nb_runner):
        """Binary struct packing across cells."""
        nb_runner.create_notebook([
            "import struct",
            textwrap.dedent("""\
                fmt = '>IHf'
                packed = struct.pack(fmt, 42, 7, 3.14)
                size = len(packed)
                print(f"size={size}")
            """),
            textwrap.dedent("""\
                val_int, val_short, val_float = struct.unpack(fmt, packed)
                print(f"int={val_int} short={val_short} float={val_float:.2f}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "size=10" in nb_runner.get_output(2)
        assert "int=42 short=7 float=3.14" in nb_runner.get_output(3)

    def test_struct_multiple_records(self, nb_runner):
        """Pack/unpack multiple records."""
        nb_runner.create_notebook([
            "import struct",
            textwrap.dedent("""\
                record_fmt = '<2sif'
                records = [(b'AB', 10, 1.5), (b'CD', 20, 2.5), (b'EF', 30, 3.5)]
                packed_data = b''.join(struct.pack(record_fmt, *r) for r in records)
                total_bytes = len(packed_data)
                print(f"total_bytes={total_bytes}")
            """),
            textwrap.dedent("""\
                rec_size = struct.calcsize(record_fmt)
                unpacked = []
                for i in range(0, len(packed_data), rec_size):
                    unpacked.append(struct.unpack(record_fmt, packed_data[i:i+rec_size]))
                print(f"count={len(unpacked)} first={unpacked[0]}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "total_bytes=" in nb_runner.get_output(2)
        out = nb_runner.get_output(3)
        assert "count=3" in out
        assert "b'AB'" in out


@pytest.mark.stress
class TestJsonSerialization:
    """Test JSON serialization patterns."""

    def test_json_custom_encoder(self, nb_runner):
        """Custom JSON encoder across cells."""
        nb_runner.create_notebook([
            "import json\nfrom datetime import datetime, date",
            textwrap.dedent("""\
                class CustomEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, (datetime, date)):
                            return obj.isoformat()
                        return super().default(obj)

                data = {'event': 'test', 'date': date(2024, 1, 15), 'count': 42}
                json_str = json.dumps(data, cls=CustomEncoder, sort_keys=True)
                print(f"json={json_str}")
            """),
            textwrap.dedent("""\
                parsed = json.loads(json_str)
                print(f"event={parsed['event']} date={parsed['date']} count={parsed['count']}")
            """),
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert '"date": "2024-01-15"' in nb_runner.get_output(2)
        assert "event=test date=2024-01-15 count=42" in nb_runner.get_output(3)

    def test_json_nested_serialization(self, nb_runner, tmp_path):
        """JSON file write/read with nested data."""
        json_path = str(tmp_path / "data.json").replace("\\", "/")
        nb_runner.create_notebook([
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
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "keys=['database', 'features', 'limits']" in nb_runner.get_output(2)
        assert "host=localhost features=3" in nb_runner.get_output(3)

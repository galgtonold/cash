"""
Batch 282: JSON serialization/deserialization interaction tests.
Tests that editing data that gets serialized to JSON and then deserialized
properly invalidates downstream cells.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.interaction, pytest.mark.stress, pytest.mark.timeout(90)]


class TestJsonSerializationInteraction:
    """Test JSON serialization patterns with cache invalidation."""

    def test_json_roundtrip_edit(self, nb_runner):
        """Editing data before JSON roundtrip should propagate."""
        nb_runner.create_notebook([
            "import json\ndata = {'name': 'Alice', 'score': 95}",
            "serialized = json.dumps(data)",
            "restored = json.loads(serialized)",
            "print(f'name={restored[\"name\"]},score={restored[\"score\"]}')",
        ])
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
        nb_runner.create_notebook([
            "import json\nconfig = {'db': {'host': 'localhost', 'port': 5432}, 'debug': True}",
            "text = json.dumps(config, indent=2)",
            "parsed = json.loads(text)",
            "info = f\"{parsed['db']['host']}:{parsed['db']['port']}\"",
            "print(f'info={info}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "info=localhost:5432" in out

        nb_runner.set_cell_source(1, "import json\nconfig = {'db': {'host': 'remote.io', 'port': 3306}, 'debug': False}")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "info=remote.io:3306" in out

    def test_json_list_of_dicts_edit(self, nb_runner):
        """Editing a list of dicts serialized as JSON."""
        nb_runner.create_notebook([
            "import json\nrecords = [{'id': 1, 'val': 10}, {'id': 2, 'val': 20}]",
            "blob = json.dumps(records)",
            "loaded = json.loads(blob)",
            "total = sum(r['val'] for r in loaded)",
            "print(f'total={total}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "total=30" in out

        nb_runner.set_cell_source(1, "import json\nrecords = [{'id': 1, 'val': 100}, {'id': 2, 'val': 200}, {'id': 3, 'val': 300}]")
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "total=600" in out

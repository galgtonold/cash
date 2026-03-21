"""
Interaction test: uuid module for unique identifier generation.
Tests uuid1, uuid4, uuid5, namespace operations,
and cross-cell UUID-based identification patterns.
"""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestUuidGeneration:
    """Test uuid generation across cells."""

    def test_uuid_ops(self, nb_runner):
        nb_runner.create_notebook([
            # Cell 1: uuid4 generation
            "import uuid\nid1 = uuid.uuid4()\nid2 = uuid.uuid4()\nprint(f'len={len(str(id1))}')\nprint(f'different={id1 != id2}')\nprint(f'version={id1.version}')",
            # Cell 2: uuid5 (deterministic)
            "ns = uuid.NAMESPACE_DNS\nid_a = uuid.uuid5(ns, 'example.com')\nid_b = uuid.uuid5(ns, 'example.com')\nprint(f'deterministic={id_a == id_b}')\nprint(f'uuid5={id_a}')",
            # Cell 3: uuid from string
            "s = str(id_a)\nparsed = uuid.UUID(s)\nprint(f'roundtrip={parsed == id_a}')\nprint(f'hex={parsed.hex}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out1 = nb_runner.get_output(1)
        assert "len=36" in out1
        assert "different=True" in out1
        assert "version=4" in out1
        out2 = nb_runner.get_output(2)
        assert "deterministic=True" in out2
        out3 = nb_runner.get_output(3)
        assert "roundtrip=True" in out3

    def test_uuid_edit(self, nb_runner):
        nb_runner.create_notebook([
            "import uuid\nid_val = uuid.uuid5(uuid.NAMESPACE_URL, 'https://example.com')\nprint(f'id={id_val}')",
            "short = str(id_val)[:8]\nprint(f'short={short}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out2a = nb_runner.get_output(2)
        assert "short=" in out2a

        # Edit URL
        nb_runner.set_cell_source(1, "import uuid\nid_val = uuid.uuid5(uuid.NAMESPACE_URL, 'https://other.com')\nprint(f'id={id_val}')")
        nb_runner.run_cells([1, 2])
        out2b = nb_runner.get_output(2)
        assert "short=" in out2b

    def test_uuid_cache(self, nb_runner):
        nb_runner.create_notebook([
            "import uuid\ndet_id = uuid.uuid5(uuid.NAMESPACE_DNS, 'test')\nhex_val = det_id.hex\nprint(f'hex={hex_val}')",
            "length = len(hex_val)\nprint(f'hex_len={length}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "hex_len=32" in nb_runner.get_output(2)

        # Re-run - cache
        nb_runner.run_all()
        assert "hex_len=32" in nb_runner.get_output(2)

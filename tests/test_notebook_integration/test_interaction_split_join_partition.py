"""Batch 469: string split join partition operations."""
import pytest

pytestmark = [pytest.mark.stress, pytest.mark.timeout(90)]


class TestStringSplitJoinPartition:
    def test_split_join_round(self, nb_runner):
        nb_runner.create_notebook([
            "text = 'hello world python'",
            "words = text.split()\njoined = '-'.join(words)\nprint(f'words={words} joined={joined}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "hello" in out
        assert "joined=hello-world-python" in out

    def test_partition(self, nb_runner):
        nb_runner.create_notebook([
            "path = 'user@host:port'",
            "user, sep, rest = path.partition('@')\nhost, sep2, port = rest.partition(':')\nprint(f'user={user} host={host} port={port}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        out = nb_runner.get_output(2)
        assert "user=user" in out
        assert "host=host" in out
        assert "port=port" in out

    def test_split_edit(self, nb_runner):
        nb_runner.create_notebook([
            "csv_line = 'a,b,c'",
            "parts = csv_line.split(',')\nprint(f'parts={parts}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "parts=['a', 'b', 'c']" in nb_runner.get_output(2)
        nb_runner.set_cell_source(1, "csv_line = 'x,y,z,w'")
        nb_runner.run_all()
        assert "parts=['x', 'y', 'z', 'w']" in nb_runner.get_output(2)

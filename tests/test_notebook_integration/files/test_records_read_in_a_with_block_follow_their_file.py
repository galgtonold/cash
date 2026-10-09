"""Records a ``with open(...)`` block reads are cached as any statement's
outputs, and still follow their file, over Run Alls and a restart.

The closed handle the block leaves bound was taken for an open stream (each
output hashed in full on every run), and after a restart every hit copied
each record to see that it could. Neither changes what the notebook shows:
the records are the file's, the handle is closed, and an edit to the file
is read on the next run.
"""

import json

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _write(path, n, shift=0):
    path.write_text(
        "".join(json.dumps({"id": i, "v": i + shift, "tags": ["a"]}) + "\n" for i in range(n)), encoding="utf-8"
    )


def test_records_read_in_a_with_block_follow_their_file(nb_runner, tmp_path):
    _write(tmp_path / "events.jsonl", 3000)
    cells = [
        "import json",
        "with open('events.jsonl') as f:\n    records = [json.loads(line) for line in f]",
        "total = sum(r['v'] for r in records)",
        "check = (len(records), total, f.closed, records[7]['tags'])",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    expected = f"(3000, {sum(range(3000))}, True, ['a'])"
    for label in ("first", "second"):
        nb_runner.run_all()
        assert nb_runner.peek("check") == expected, f"{label} Run All"

    _write(tmp_path / "events.jsonl", 3000, shift=1)
    nb_runner.run_all()
    edited = f"(3000, {sum(range(3000)) + 3000}, True, ['a'])"
    assert nb_runner.peek("check") == edited, "after the file changed"

    nb_runner.restart()
    nb_runner._init_cash()
    for label in ("after a restart", "again after a restart"):
        nb_runner.run_all()
        assert nb_runner.peek("check") == edited, label

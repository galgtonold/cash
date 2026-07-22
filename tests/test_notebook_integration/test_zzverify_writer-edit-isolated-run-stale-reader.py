"""Minimal repro (verification of probe finding writer-edit-isolated-run-stale-reader).

Editing a file-WRITING cell and then re-running only the READER cell serves the
stale pre-edit file content. Root cause: file writes are untracked
(src/cash/notebook/file_tracker.py, documented limitation), so the upstream
simulation has no writer->reader dependency edge, never re-executes the edited
writer, and the reader (re-)reads / restores against the unchanged on-disk file.

The variable-edge equivalent (edit `a = 2` -> `a = 5`, run only the consumer)
is core promised behavior and passes (test_cross_cell_edit_propagation.py);
run_all after the same edit also recovers. Only the file-mediated edge on the
isolated-run path is broken.
"""

import pytest

pytestmark = [pytest.mark.timeout(90)]


def test_writer_edit_then_isolated_reader_rerun_serves_stale_file(nb_runner, tmp_path):
    p = str(tmp_path / "data.txt").replace("\\", "/")
    writer_v1 = (
        f"with open('{p}', 'w') as f:\n"
        "    f.write('1,2,3')\n"
        "print('wrote v1')"
    )
    reader = (
        f"with open('{p}') as f:\n"
        "    body = f.read()\n"
        "print('body =', body)"
    )
    nb_runner.create_notebook([writer_v1, reader])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "body = 1,2,3" in nb_runner.get_output(2)

    # Edit the writer: different content AND byte size (rules out CAS-10
    # mtime-granularity timing entirely -- the file is simply never rewritten).
    writer_v2 = (
        f"with open('{p}', 'w') as f:\n"
        "    f.write('10,20,30,40')\n"
        "print('wrote v2')"
    )
    nb_runner.set_cell_source(1, writer_v2)

    # Isolated re-run of ONLY the reader cell (what a user does after editing
    # the writer). Upstream simulation must re-execute the edited writer so the
    # reader sees the new file -- the promised behavior for variable edges.
    nb_runner.run_cell(2)
    out = nb_runner.get_output(2)
    assert "body = 10,20,30,40" in out, (
        f"Reader cell served STALE file content after the writer cell was "
        f"edited (writer never re-executed by upstream simulation; file writes "
        f"are untracked so there is no writer->reader edge). Got: {out!r}"
    )

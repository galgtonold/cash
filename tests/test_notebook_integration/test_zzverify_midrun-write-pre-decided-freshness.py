"""Minimal repro (verification of probe finding midrun-write-pre-decided-freshness).

Sharper variant of writer-edit-isolated-run-stale-reader: the consumer cell
references BOTH the writer's variable (``payload``) and the reader's variable
(``loaded``), so the upstream simulation is FORCED to re-execute the edited
writer cell during the isolated run. The writer executes and (maybe) rewrites
the file mid-run — yet the reader's ``loaded`` is still restored from the
pre-edit cache, producing an internally inconsistent namespace:
``payload`` is the new 4-element object while ``loaded`` is the old 3-element
unpickle of it. No plain-Python execution order can produce that state.

The test additionally inspects the on-disk pickle after the isolated run to
discriminate two candidate mechanisms:
  (a) file WAS rewritten mid-run -> reader freshness was decided against the
      pre-write file state and never re-checked (pre-decided freshness), or
  (b) file was NOT rewritten -> the sim re-ran the assignment statement but
      skipped the side-effect-only ``with``-dump statement (write channel
      dropped even for a re-executed cell).
Either way the user-visible result is the same inconsistent namespace.
"""

import pickle

import pytest

pytestmark = [pytest.mark.timeout(90)]


def test_reexecuted_writer_midrun_reader_still_restores_stale(nb_runner, tmp_path):
    pkl = tmp_path / "both.pkl"
    p = str(pkl).replace("\\", "/")
    writer_v1 = (
        "import pickle\n"
        "payload = {'nums': [1, 2, 3]}\n"
        f"with open('{p}', 'wb') as f:\n"
        "    pickle.dump(payload, f)\n"
        "print('dumped', len(payload['nums']))"
    )
    reader = (
        "import pickle\n"
        f"with open('{p}', 'rb') as f:\n"
        "    loaded = pickle.load(f)\n"
        "print('loaded =', loaded['nums'])"
    )
    # Consumer needs BOTH payload (writer var) and loaded (reader var): the
    # upstream simulation cannot satisfy the consumer without re-executing
    # the edited writer cell.
    consumer = "print('combo =', sum(loaded['nums']), len(payload['nums']))"
    nb_runner.create_notebook([writer_v1, reader, consumer])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "combo = 6 3" in nb_runner.get_output(3)

    # Edit ONLY the payload literal (different length -> different pickle size,
    # so CAS-10 mtime/size granularity is not in play).
    writer_v2 = writer_v1.replace("[1, 2, 3]", "[10, 20, 30, 40]")
    nb_runner.set_cell_source(1, writer_v2)

    # Isolated run of ONLY the consumer.
    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)

    # Mechanism discriminator: what does the file on disk hold NOW?
    with open(pkl, "rb") as f:
        on_disk = pickle.load(f)["nums"]

    assert "combo = 100 4" in out, (
        f"INCONSISTENT NAMESPACE after isolated downstream run: writer WAS "
        f"re-executed (len(payload)==4 in output) but reader's 'loaded' was "
        f"restored stale. Cell output: {out!r}. On-disk pickle now holds "
        f"{on_disk!r} -> "
        + (
            "file WAS rewritten mid-run; reader freshness was pre-decided "
            "against the pre-write state and never re-checked."
            if on_disk == [10, 20, 30, 40]
            else "file was NOT rewritten; the sim skipped the side-effect-only "
            "dump statement of the re-executed writer cell."
        )
    )

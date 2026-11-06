"""What is built on a value kept in memory only restores after a restart.

``faces = fetch_lfw_people(...)`` read its files quickly into a value too
big for the time it took, so it was kept in memory only; ``X_train = ...``
(14 s) and ``grid.fit(...)`` (61 s) below it were stored on disk. Their
entries referred to the reader's entry for the records of the files their
values were built from (``cash.notebook.file_records``); after a restart
that key held other records, or none, and each ran again, "file changed:
<cash: file records of ...>", though no file had changed. An
entry below one kept in memory only now holds those records itself, and a
real change to the file still runs it again.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

CELLS = [
    "import cash\n%load_ext cash\n%cash_badge print\n%cash_on",
    "import time\nimport numpy as np",
    # Quick, and 40 MB: not worth the disk, kept in memory only.
    "time.sleep(0.15)\nraw = np.tile(np.frombuffer(open('data.bin', 'rb').read(), dtype=np.uint8), 40_000)",
    "s = (time.sleep(0.6), int(raw[::1000].sum()))[1]\nprint('S', s)",
    "t = (time.sleep(0.6), s * 2)[1]\nprint('T', t)",
]


_DISK = "__import__('cash._active', fromlist=['x']).default_cash().backend.backends[1]"
# What the disk holds of the entry below the reader: whether it refers to
# the reader's entry, and whether it holds the record of the file itself.
_BELOW = (
    f"[(any(r[0] == {{k}} for r in m.get('file_refs') or ()), any(p.endswith('data.bin') for p in "
    "m.get('file_dependencies') or ())) "
    f"for d in [{_DISK}] for e in d.list_entries() for m in [d.peek_metadata(e['key']) or dict()] "
    "if m.get('code', '').startswith('s = ')]"
)


def _write(path, byte: int) -> None:
    path.write_bytes(bytes([byte]) * 1000)


def _first_run(nb_runner):
    _write(nb_runner.work_dir / "data.bin", 1)
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "S 40000" in nb_runner.get_output(4), nb_runner.get_raw_output(4)
    stored = nb_runner.get_raw_output(3)
    assert "-> RAM\n" in stored + "\n" and "raw = " in stored, f"control: the reader is kept in memory only\n{stored}"
    nb_runner.settle_writes()
    reader = nb_runner.peek(
        f"[e['key'] for d in [{_DISK}] for e in d.list_entries() if e['key'].startswith('stmt:') "
        "and (d.peek_metadata(e['key']) or dict()).get('code', '').startswith('raw = ')]"
    )
    assert reader.startswith("['stmt:"), reader
    below = nb_runner.peek(_BELOW.replace("{k}", reader[1:-1]))
    assert below == "[(False, True)]", (
        f"the entry below refers to the reader's, or holds no record of its file: {below}"
    )


def test_the_cells_below_restore_and_a_changed_file_runs_them_again(nb_runner):
    _first_run(nb_runner)
    nb_runner.settle_writes()

    nb_runner.restart()
    nb_runner.run_all()
    for cell, line in ((4, "S 40000"), (5, "T 80000")):
        out = nb_runner.get_output(cell)
        assert line in out and shows_cached(out), nb_runner.get_raw_output(3) + "\n" + nb_runner.get_raw_output(cell)

    nb_runner.settle_writes()
    nb_runner.restart()
    _write(nb_runner.work_dir / "data.bin", 2)
    nb_runner.run_all()
    for cell, line in ((4, "S 80000"), (5, "T 160000")):
        out = nb_runner.get_output(cell)
        assert line in out and not shows_cached(out), nb_runner.get_raw_output(cell)

"""A statement that reads audio through soundfile runs again once the file changes.

soundfile reads through libsndfile in C: no ``open()`` audit event, and soundfile
was not in the C-reader registry, so the notebook stored the statement with no
file dependency and an edited .wav restored the old value.
"""

from __future__ import annotations

import os

import pytest

from tests._cell_driver import run_cash_cell

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")


def _wav(path, value):
    sf.write(str(path), np.full(100, value), 8000)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))


@pytest.mark.parametrize(
    "read",
    [
        "round(float(sf.read(P)[0].sum()))",
        "round(float(sf.SoundFile(P).read().sum()))",
        "round(float(sum(b.sum() for b in sf.blocks(P, blocksize=10))))",
    ],
    ids=["read", "SoundFile", "blocks"],
)
def test_an_edited_wav_reruns_its_reader(cash_magics, mock_shell, tmp_path, read):
    path = tmp_path / "a.wav"
    _wav(path, 0.0)
    run_cash_cell(cash_magics, f"import soundfile as sf, time\nP = {str(path)!r}")
    run_cash_cell(cash_magics, "def load():\n    time.sleep(0.2)\n    return " + read)
    run_cash_cell(cash_magics, "total = load()")
    assert mock_shell.user_ns["total"] == 0

    _wav(path, 0.5)
    run_cash_cell(cash_magics, "total = load()")
    assert mock_shell.user_ns["total"] == 50, "the edited .wav restored the old value"

"""soundfile's readers are tracked: an edited audio file recomputes.

soundfile reads through libsndfile in C, so nothing passes through a patched
``open()`` and no audit event reports the read. A cached function over
``sf.read``, ``sf.SoundFile`` or ``sf.blocks`` recorded no file dependency:
after the .wav was rewritten it returned the old file's answer with no warning.
"""

from __future__ import annotations

import os

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

pytestmark = pytest.mark.core


def _wav(path, value):
    sf.write(str(path), np.full(100, value), 8000)
    st = os.stat(path)  # an older mtime, so the rewrite below is a change by any rule
    os.utime(path, ns=(st.st_atime_ns - 3600 * 10**9, st.st_mtime_ns - 3600 * 10**9))


def _via_read(p):
    data, _ = sf.read(p)
    return round(float(data.sum()))


def _via_class(p):
    with sf.SoundFile(p) as f:
        return round(float(f.read().sum()))


def _via_blocks(p):
    return round(float(sum(b.sum() for b in sf.blocks(p, blocksize=10))))


def _via_info(p):
    return sf.info(p).frames


READERS = {"read": _via_read, "SoundFile": _via_class, "blocks": _via_blocks}


@pytest.mark.parametrize("reader", sorted(READERS))
def test_an_edited_audio_file_recomputes(disk_cash, tmp_path, reader):
    read = READERS[reader]
    path = tmp_path / "a.wav"
    _wav(path, 0.0)
    runs = []

    @disk_cash.cache(assume_safe=True)
    def total(p):
        runs.append(1)
        return read(p)

    assert total(str(path)) == 0
    assert total(str(path)) == 0
    assert len(runs) == 1, "an unchanged file recomputed"

    _wav(path, 0.5)
    assert total(str(path)) == 50, f"sf.{reader} served the old file's answer"
    assert len(runs) == 2


def test_a_longer_file_changes_what_info_reports(disk_cash, tmp_path):
    path = tmp_path / "a.wav"
    _wav(path, 0.0)

    @disk_cash.cache(assume_safe=True)
    def frames(p):
        return _via_info(p)

    assert frames(str(path)) == 100
    sf.write(str(path), np.zeros(250), 8000)
    assert frames(str(path)) == 250


def test_only_a_soundfile_opened_to_read_is_a_dependency(tmp_path):
    """``SoundFile(p, 'w')`` makes an output, not an input; ``'r'`` reads one."""
    from cash.tracking.file_tracker import FileAccessTracker

    src = tmp_path / "in.wav"
    _wav(src, 0.0)
    out = tmp_path / "out.wav"
    with FileAccessTracker() as tracker:
        with sf.SoundFile(str(src)) as f:
            data = f.read()
        with sf.SoundFile(str(out), "w", samplerate=8000, channels=1) as f:
            f.write(data)
    accessed = {os.path.normcase(os.path.realpath(p)) for p in tracker.get_accessed_files()}
    assert os.path.normcase(os.path.realpath(src)) in accessed
    assert os.path.normcase(os.path.realpath(out)) not in accessed

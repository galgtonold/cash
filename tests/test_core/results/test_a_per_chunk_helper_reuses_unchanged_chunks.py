"""The shape ``CACHE-PER-ITEM-RERUN`` advises works: a ``@cash.cache``
helper called per chunk of rows computes only the chunks that are new or
changed after rows are appended or a batch in the middle is redone, and the
combined result equals computing everything again.
"""

from __future__ import annotations

import os

import numpy as np

CHUNK = 250


def _counting_prep(cash, log: str):
    @cash.cache
    def prep_chunk(block):
        fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
        os.write(fd, b"x")
        os.close(fd)
        return block.astype(np.float32) / 255.0

    return prep_chunk


def _computes(log: str) -> int:
    return os.path.getsize(log) if os.path.exists(log) else 0


def _run(prep, imgs):
    return np.concatenate([prep(imgs[s : s + CHUNK]) for s in range(0, len(imgs), CHUNK)])


def test_only_new_and_changed_chunks_are_computed(disk_cash, tmp_path):
    log = str(tmp_path / "computes.log")
    prep = _counting_prep(disk_cash, log)
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 255, (3000, 8, 8), dtype=np.uint8)

    first = _run(prep, imgs)
    assert _computes(log) == 12, "control: every chunk computed once"

    grown = np.concatenate([imgs, rng.integers(0, 255, (400, 8, 8), dtype=np.uint8)])
    after_append = _run(prep, grown)
    assert _computes(log) == 12 + 2, "only the two chunks of the 400 appended rows"
    np.testing.assert_array_equal(after_append, grown.astype(np.float32) / 255.0)
    np.testing.assert_array_equal(after_append[:3000], first)

    redone = grown.copy()
    redone[1000:1250] = 0  # one batch in the middle exported again
    after_redo = _run(prep, redone)
    assert _computes(log) == 14 + 1, "only the redone chunk"
    np.testing.assert_array_equal(after_redo, redone.astype(np.float32) / 255.0)

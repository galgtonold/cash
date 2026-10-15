"""What a notebook's cache holds on disk, counted the way the store keeps it.

An entry no longer carries every byte of its value: a buffer of at least
`cash.backends.chunk_store.CHUNK_MIN_BYTES` (an array's data, a frame's
column) lives once per cache folder in ``chunks/<digest>.chunk``, and the
entry's metadata names it. Counting only ``*.entry`` files therefore sees a
48 MB array as a few KB. These helpers count an entry together with the
chunks it holds.
"""

from __future__ import annotations

import os

from cash.backends.cache_dir import chunk_files
from cash.backends.entry_format import ENTRY_SUFFIX

#: A cell to run in the kernel: prints ``DATA_FILES=<entries>`` and
#: ``MAX_DATA_BYTES=<bytes of the biggest entry, with its chunks>`` and
#: ``MAX_DATA_SECONDS=<the compute that entry recorded keeping saves>`` for the
#: ``.cash`` folder of the kernel's own cwd. A chunk the entry names but that
#: is not on disk counts nothing, so the number is what a restart can read.
ENTRY_BYTES_REPORT = (
    "import os, glob\n"
    "from cash.backends._base import store_seconds\n"
    "from cash.backends.entry_format import CHUNKS_FIELD, read_entry\n"
    "_cash = os.path.abspath('.cash')\n"
    "def _held(_path):\n"
    "    _meta, _ = read_entry(_path, with_payload=False)\n"
    "    _n = os.path.getsize(_path)\n"
    "    for _name, _size in _meta.get(CHUNKS_FIELD) or ():\n"
    "        _chunk = os.path.join(_cash, 'chunks', _name)\n"
    "        if os.path.exists(_chunk):\n"
    "            _n += os.path.getsize(_chunk)\n"
    "    return _n, store_seconds(_meta)\n"
    "_datas = glob.glob(os.path.join(_cash, '*.entry'))\n"
    "_maxb, _maxs = max((_held(f) for f in _datas), default=(0, 0.0))\n"
    "print(f'DATA_FILES={len(_datas)} MAX_DATA_BYTES={_maxb} MAX_DATA_SECONDS={_maxs:.3f}')"
)


def cache_bytes(cache_dir: str | os.PathLike) -> int:
    """Bytes of every entry in *cache_dir* plus every chunk in its store
    (each chunk once, however many entries hold it)."""
    root = os.fspath(cache_dir)
    entries = sum(e.stat().st_size for e in os.scandir(root) if e.name.endswith(ENTRY_SUFFIX))
    return entries + sum(os.path.getsize(path) for path in chunk_files(root))

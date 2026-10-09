"""How the file backend compresses an entry's payload (``compress=True``).

The codec is the fastest good one this Python has: zstd (the standard
library's ``compression.zstd``, Python 3.14+) at level 3, else zlib at level
1. Both run at hundreds of MB/s and land within a few percent of gzip's
smallest files; gzip at its default level 9 took 30 s for a 59 MB table.

A payload that does not shrink is stored as it is: the codec runs on its
first `SAMPLE_BYTES` first, and if that saves less than `MIN_SAVING` the
whole payload is not worth compressing (random floats, already-compressed
images), so reading it back costs nothing extra either.

The metadata's ``compressed`` field names the codec that wrote the payload
(``"zstd"``, ``"zlib"``), or is false for a payload stored as it is. Entries
written before codecs were named carry ``True``, which means gzip, and still
read. A codec this Python lacks (a zstd entry read by Python 3.13) reads as a
miss: the value is computed again, never guessed at.
"""

from __future__ import annotations

import gzip
import zlib
from typing import Any

__all__ = ["CODEC", "SAMPLE_BYTES", "MIN_SAVING", "compress", "decompress", "UnknownCodec"]

try:  # Python 3.14+
    from compression import zstd as _zstd
except ImportError:  # pragma: no cover - depends on the Python version
    _zstd = None

#: zstd's own default level: 3 is about as fast as 1 on cash's payloads and a
#: little smaller.
ZSTD_LEVEL = 3
#: zlib's fastest level: about 30x faster than gzip's default 9 on a pickled
#: table, for a file 2% larger.
ZLIB_LEVEL = 1

#: The codec new entries are written with.
CODEC = "zstd" if _zstd is not None else "zlib"

#: How much of a payload is compressed to decide whether the rest is worth it.
SAMPLE_BYTES = 1 << 20
#: Below this saving on the sample, the payload is stored uncompressed.
MIN_SAVING = 0.10


class UnknownCodec(ValueError):
    """The payload was written with a codec this Python cannot read."""


def _compress_with(codec: str, data: Any) -> bytes:
    if codec == "zstd":
        return _zstd.compress(data, level=ZSTD_LEVEL)
    return zlib.compress(data, ZLIB_LEVEL)


def compress(data: Any) -> tuple[str | None, Any]:
    """``(codec, compressed bytes)``, or ``(None, data)`` when *data* does not
    shrink by `MIN_SAVING` (judged on its first `SAMPLE_BYTES`)."""
    view = memoryview(data).cast("B")
    sample = view[:SAMPLE_BYTES]
    if len(sample) == 0:
        return None, data
    packed = _compress_with(CODEC, sample)
    if len(packed) > (1.0 - MIN_SAVING) * len(sample):
        return None, data
    if len(view) > len(sample):
        packed = _compress_with(CODEC, view)
    if len(packed) >= len(view):
        return None, data
    return CODEC, packed


def decompress(codec: Any, data: bytes) -> bytes:
    """The payload *codec* (the metadata's ``compressed`` field) compressed.

    Raises `UnknownCodec` for a codec this Python cannot read, and the
    codec's own error (``zlib.error``, ``OSError``, ``EOFError``, a zstd
    error) for bytes it cannot inflate.
    """
    if codec is True or codec == "gzip":
        return gzip.decompress(data)
    if codec == "zlib":
        return zlib.decompress(data)
    if codec == "zstd":
        if _zstd is None:
            raise UnknownCodec("zstd needs Python 3.14 or newer")
        return _zstd.decompress(data)
    raise UnknownCodec(f"unknown codec {codec!r}")

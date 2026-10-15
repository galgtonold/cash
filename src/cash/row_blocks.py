"""Block fingerprints of a value made of rows, to say how much of it changed
between two runs (``CACHE-PER-ITEM-RERUN``).

A step that recomputes over a whole collection when only part of it changed
(400 new images appended to 3,000, one season added to fifty) cannot be
reused by cash: the input is one value, and it changed. The user can fix that
by doing the work through a cached helper per item or per chunk; to say so,
cash needs to know how much of the input is the same as last time. That is
what this module answers, cheaply and without keeping any values.

A fingerprint cuts the rows into at most :data:`MAX_BLOCKS` blocks of
``block`` rows and keeps a short digest of each block, taken with cash's own
content hashing (:func:`~cash.content_hashers.builtin_hash`) on the row
slice. The next run hashes its input on the same grid, so an appended row,
or a changed block in the middle, shows up as the blocks whose digests
differ. Values that are made of rows: a numpy array of one or more axes,
a pandas DataFrame or Series, a polars DataFrame or Series, and a list or
tuple of numpy arrays (one row per item).

Advice only: nothing here reaches a key or decides what is served.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ._clock import perf_counter as _perf_counter
from .bulk_digest import PARALLEL_MIN_BYTES, SCHEME, hashing_threads, leaf_hasher
from .content_hashers import builtin_hash, builtin_hash_family, raw_view

__all__ = [
    "MAX_BLOCKS",
    "MIN_ROWS",
    "HashBudgetSpent",
    "RowBlocks",
    "RowChange",
    "compare",
    "describe_ranges",
    "fingerprint",
    "row_count",
]

#: The most blocks a first fingerprint cuts a value into. Fine enough to
#: place a change to within 1/64 of the rows, small enough to keep (about
#: a kilobyte per input).
MAX_BLOCKS = 64
#: A grid kept from an earlier run is reused while the value has at most
#: this many blocks on it; past that (the value grew fourfold) it is cut
#: afresh, and that run compares nothing.
_MAX_REUSED_BLOCKS = 4 * MAX_BLOCKS
#: Fewer rows than this are not worth cutting into blocks.
MIN_ROWS = 16
#: The scheme of a record written before records named theirs.
_SHA256_SCHEME = "cash-bulk-sha256-tree-1"
#: Hex digits kept of each block's digest: 64 bits, plenty to tell two
#: blocks of one value apart.
_DIGEST_CHARS = 16


class HashBudgetSpent(Exception):
    """Hashing the blocks would take longer than the caller allows."""


def row_count(value: Any) -> tuple[str, int] | None:
    """``(kind, rows)`` for a value made of rows, or None.

    Only the types cash hashes by content, and a list or tuple whose items
    are all numpy arrays. Reads lengths and types only."""
    family = builtin_hash_family(type(value))
    try:
        if family == "numpy":
            if value.ndim < 1 or value.dtype.hasobject:
                return None
            return "ndarray", int(value.shape[0])
        if family == "pandas":
            return "pandas", len(value)
        if family == "polars":
            if type(value).__name__ not in ("DataFrame", "Series"):
                return None  # a LazyFrame has no rows yet
            return "polars", len(value)
    except (TypeError, AttributeError):
        return None
    if type(value) in (list, tuple) and len(value) >= MIN_ROWS:
        if all(builtin_hash_family(type(item)) == "numpy" for item in value):
            return "arrays", len(value)
    return None


def _slicer(kind: str, value: Any) -> Callable[[int, int], Any]:
    if kind == "polars":
        return lambda a, b: value.slice(a, b - a)
    return lambda a, b: value[a:b]


#: Seconds per row and column to hash a frame's rows, measured on pandas
#: 3 (900k rows): a numeric column about 10 ns, a string or object column
#: about 40 ns. Rounded up, to skip a frame that would take too long
#: before starting rather than after.
_NS_PER_NUMERIC_CELL = 12
_NS_PER_OTHER_CELL = 50


def _frame_seconds(value: Any) -> float:
    """About how long hashing the rows of a pandas frame or series takes."""
    dtypes = list(value.dtypes) if hasattr(value, "columns") else [value.dtype]
    dtypes.append(value.index.dtype)
    ns = sum(_NS_PER_NUMERIC_CELL if getattr(dt, "kind", "O") in "biufcmM" else _NS_PER_OTHER_CELL for dt in dtypes)
    return len(value) * ns * 1e-9


def _pandas_rows(value: Any) -> Callable[[int, int], Any]:
    """Rows of a pandas frame or series as ``(schema, row hashes)`` slices.

    One vectorised pass (``hash_pandas_object``, which cash's frame hash
    is built on) gives a 64-bit hash per row, index included; the schema
    (column names, dtypes) comes from cash's hash of the empty frame. Far
    cheaper than hashing 64 frame slices one by one."""
    import pandas as pd

    schema = (builtin_hash(value.iloc[:0]) or "?").encode()
    hashes = pd.util.hash_pandas_object(value, index=True).to_numpy()
    return lambda a, b: (schema, hashes[a:b])


def _block_digest(kind: str, part: Any) -> str:
    """The digest of one block of rows, *part*."""
    if kind == "pandas-rows":
        schema, hashes = part
        return hashlib.sha256(schema + hashes.tobytes()).hexdigest()[:_DIGEST_CHARS]
    if kind == "arrays":
        h = hashlib.sha256()
        for item in part:
            h.update((builtin_hash(item) or "?").encode())
        return h.hexdigest()[:_DIGEST_CHARS]
    digest = builtin_hash(part)
    if digest is None:
        raise ValueError("block not hashable")
    return digest[:_DIGEST_CHARS]


@dataclass(frozen=True)
class RowBlocks:
    """The fingerprint of one value made of rows: no values, only digests.

    ``digests[i]`` covers rows ``[i * block, min((i + 1) * block, rows))``.
    """

    kind: str
    rows: int
    block: int
    digests: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "rows": self.rows,
            "block": self.block,
            "digests": list(self.digests),
            "scheme": SCHEME,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RowBlocks | None:
        """The fingerprint *data* holds, or None for one this process cannot
        compare with its own: unreadable, or taken with another digest
        scheme (`cash.bulk_digest.SCHEME`; a record without one is from a
        build that had only SHA-256)."""
        try:
            if data.get("scheme", _SHA256_SCHEME) != SCHEME:
                return None
            return cls(str(data["kind"]), int(data["rows"]), int(data["block"]), tuple(map(str, data["digests"])))
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class RowChange:
    """How a value differs from the one an earlier fingerprint was taken of."""

    rows: int
    #: Rows that differ or are new, as half-open ``(start, stop)`` ranges.
    changed: tuple[tuple[int, int], ...]

    @property
    def changed_rows(self) -> int:
        return sum(stop - start for start, stop in self.changed)

    @property
    def unchanged_share(self) -> float:
        return 1.0 - self.changed_rows / self.rows if self.rows else 0.0


def fingerprint(
    value: Any, *, like: RowBlocks | None = None, deadline: float | None = None
) -> tuple[RowBlocks, str | None] | None:
    """The block fingerprint of *value*, or None when it is not made of rows
    or a block cannot be hashed.

    With *like*, an earlier fingerprint of the same input, the blocks are cut
    on its grid, so the two compare block by block, and the second item is
    the digest of the rows that were *like*'s last, partial block, when the
    value has more rows now (else None). Raises :class:`HashBudgetSpent` once
    *deadline* (a ``perf_counter`` time) has passed, or as soon as the first
    block says the whole would not finish by then.
    """
    shape = row_count(value)
    if shape is None:
        return None
    kind, rows = shape
    if rows < MIN_ROWS:
        return None
    block = 0
    if like is not None and like.kind == kind and like.block > 0 and rows <= like.block * _MAX_REUSED_BLOCKS:
        block = like.block
    else:
        like = None
    if not block:
        block = math.ceil(rows / MAX_BLOCKS)
    count = math.ceil(rows / block)
    digests: list[str] = []
    try:
        if kind == "ndarray" and _plain_buffer(value):
            return _array_fingerprint(value, rows, block, like, deadline)
        if kind == "pandas":
            if deadline is not None and _perf_counter() + _frame_seconds(value) > deadline:
                raise HashBudgetSpent
            # One pass that cannot stop half way: judged before it starts,
            # and finished once started (what it cost is spent either way).
            part, digest_kind = _pandas_rows(value), "pandas-rows"
            deadline = None
        else:
            part, digest_kind = _slicer(kind, value), kind
        for i in range(count):
            started = _perf_counter()
            digests.append(_block_digest(digest_kind, part(i * block, min((i + 1) * block, rows))))
            if deadline is not None:
                now = _perf_counter()
                if now > deadline or (i == 0 and now + (now - started) * (count - 1) > deadline):
                    raise HashBudgetSpent
        cut = None
        if like is not None and like.rows % block and rows > like.rows:
            first = like.rows // block * block
            cut = _block_digest(digest_kind, part(first, like.rows))
    except (ImportError, TypeError, ValueError, AttributeError, IndexError, MemoryError):
        return None
    return RowBlocks(kind, rows, block, tuple(digests)), cut


def _plain_buffer(value: Any) -> bool:
    """A C-contiguous array of plain items: its rows are byte ranges."""
    return bool(value.flags.c_contiguous) and value.dtype.fields is None and value.dtype.kind != "O"


def _array_fingerprint(
    value: Any, rows: int, block: int, like: RowBlocks | None, deadline: float | None
) -> tuple[RowBlocks, str | None]:
    """:func:`fingerprint` of a C-contiguous array: each block is a byte
    range of its buffer, digested behind the row shape and dtype, the
    blocks on several threads when the array is big (hashlib and blake3
    let go of the GIL while they hash). The digest is the leaf algorithm's
    (`cash.bulk_digest.leaf_hasher`)."""
    data = memoryview(raw_view(value)).cast("B")
    width = len(data) // rows if rows else 0
    prefix = f"{value.shape[1:]}:{value.dtype}:".encode()

    def digest(a: int, b: int) -> str:
        h = leaf_hasher(prefix)
        h.update(data[a * width : b * width])
        return h.hexdigest()[:_DIGEST_CHARS]

    count = math.ceil(rows / block)
    spans = [(i * block, min((i + 1) * block, rows)) for i in range(count)]
    started = _perf_counter()
    first = digest(*spans[0])
    threads = hashing_threads() if len(data) >= PARALLEL_MIN_BYTES else 1
    if deadline is not None:
        now = _perf_counter()
        if now + (now - started) * (count - 1) / threads > deadline:
            raise HashBudgetSpent
    if threads > 1 and count > 2:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            rest = list(pool.map(lambda span: digest(*span), spans[1:]))
    else:
        rest = [digest(*span) for span in spans[1:]]
    cut = None
    if like is not None and like.rows % block and rows > like.rows:
        cut = digest(like.rows // block * block, like.rows)
    return RowBlocks("ndarray", rows, block, (first, *rest)), cut


def compare(old: RowBlocks, new: RowBlocks, cut: str | None = None) -> RowChange | None:
    """Which rows of *new* differ from *old*, or None when the two were not
    cut on one grid (another kind of value, or the grid was cut afresh).

    A block counts as unchanged when its digest is the one *old* has at the
    same place. The rows of *old*'s last, partial block count as unchanged
    when *cut* (see :func:`fingerprint`) matches it. Everything else of
    *new*, including every row past *old*'s end, has changed.
    """
    if old.kind != new.kind or old.block != new.block:
        return None
    block = new.block
    same: list[tuple[int, int]] = []
    for i, digest in enumerate(old.digests):
        start = i * block
        old_stop = min(start + block, old.rows)
        new_stop = min(start + block, new.rows)
        if start >= new.rows:
            break
        if old_stop == new_stop:
            if i < len(new.digests) and new.digests[i] == digest:
                same.append((start, new_stop))
        elif old_stop < new_stop and cut is not None and cut == digest:
            same.append((start, old_stop))
    changed: list[tuple[int, int]] = []
    at = 0
    for start, stop in same:
        if start > at:
            _extend(changed, at, start)
        at = stop
    if at < new.rows:
        _extend(changed, at, new.rows)
    return RowChange(new.rows, tuple(changed))


def _extend(ranges: list[tuple[int, int]], start: int, stop: int) -> None:
    if ranges and ranges[-1][1] == start:
        ranges[-1] = (ranges[-1][0], stop)
    else:
        ranges.append((start, stop))


def describe_ranges(ranges: tuple[tuple[int, int], ...], most: int = 3) -> str:
    """``rows 3000-3399`` or ``rows 0-46, 2961-3399``; at most *most* ranges,
    then ``...``."""
    parts = [f"{a}-{b - 1}" if b - a > 1 else f"{a}" for a, b in ranges[:most]]
    if len(ranges) > most:
        parts.append("...")
    word = "row" if len(ranges) == 1 and ranges[0][1] - ranges[0][0] == 1 else "rows"
    return f"{word} {', '.join(parts)}"

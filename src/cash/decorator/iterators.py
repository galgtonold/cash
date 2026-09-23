"""The iterators a cached generator hands back: one that streams a miss
through while it is stored, one that replays a hit chunk by chunk."""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

from ..exceptions import CacheBackendError


class StreamingCachedIterator:
    """Passes the producer's items through as they arrive, caching at the end.

    Returned on a MISS. `@cash.cache` should not change how a function
    behaves, and for a generator it used to: cash drained the whole thing
    before returning anything, so a streamed response arrived all at once
    after the full latency. Measured on a token stream -- 494ms to first item
    uncached, 2444ms cached, the entire completion in one go.

    Same surface as the replay iterator, deliberately: `send` and `throw`
    raise, because a cached generator cannot support them on the hit either.
    """

    __slots__ = ("_gen",)

    def __init__(self, gen):
        self._gen = gen

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._gen)

    def close(self):
        """Abandon the stream. Nothing is cached -- see `_stream_and_store`."""
        self._gen.close()

    def send(self, value):
        raise AttributeError(
            "cached generator: .send() is not supported. If you need send() semantics, the function cannot be cached."
        )

    def throw(self, *args, **kwargs):
        raise AttributeError(
            "cached generator: .throw() is not supported. If you need throw() semantics, the function cannot be cached."
        )


class ChunkedCachedIterator:
    """Lazy iterator that reads cached chunks from the backend on demand.

    Used by `Cash.cache` for iterator-returning functions whose
    output spans multiple backend keys. Each chunk is fetched only
    when the user iterates into it; chunks the user never reaches are
    never read. The retrieval is RAM-bounded by chunk size.

    The class satisfies the iterator protocol (``iter(x) is x``,
    ``__next__``, ``close``); generator-specific methods (``send``,
    ``throw``) raise ``AttributeError`` - the cached iterator is a
    replay of stored values, not a coroutine.

    Args:
        cash: The owning `Cash` instance (used for backend access).
        cache_key: The canonical key under which the manifest is stored.
            Chunk keys are derived as ``f"{cache_key}:chunk_{i}"``.
        n_chunks: Total chunk count, taken from the manifest at construction.

    A chunk can go while the caller is still reading: another process clears
    or rewrites the entry, or the RAM tier evicts it. ``_chunks_are_intact``
    is checked at lookup, which is before that -- so a lost chunk used to end
    the iteration, and the caller got a silent PREFIX (100 of 1000 items,
    found attacking the decorator before round 26). The rest is recomputed
    from *recompute* instead, skipping what was already yielded; with no way
    to recompute, the loss is raised. A truncated answer is worse than a slow
    one.
    """

    __slots__ = (
        "_cash",
        "_cache_key",
        "_n_chunks",
        "_chunk_index",
        "_current_chunk_iter",
        "_closed",
        "_recompute",
        "_yielded",
    )

    def __init__(self, cash: Any, cache_key: str, n_chunks: int, recompute: Callable[[], Any] | None = None):
        self._cash = cash
        self._cache_key = cache_key
        self._n_chunks = n_chunks
        self._chunk_index = 0
        self._current_chunk_iter = None
        self._closed = False
        self._recompute = recompute
        self._yielded = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        while True:
            if self._current_chunk_iter is not None:
                try:
                    item = next(self._current_chunk_iter)
                except StopIteration:
                    self._current_chunk_iter = None
                    # Fall through to load the next chunk.
                else:
                    self._yielded += 1
                    return item
            if self._chunk_index >= self._n_chunks:
                raise StopIteration
            chunk_key = f"{self._cache_key}:chunk_{self._chunk_index}"
            _, chunk = self._cash.backend.get(chunk_key)
            self._chunk_index += 1
            if chunk is None:
                # The chunk went while the caller was reading (see the class
                # docstring). Finish the run from the function itself.
                if self._recompute is None:
                    raise CacheBackendError(
                        f"a chunk of the cached result for {self._cache_key} is gone "
                        f"after {self._yielded} items; the rest cannot be read"
                    )
                fresh = iter(self._recompute())
                for _ in range(self._yielded):
                    next(fresh, None)
                self._current_chunk_iter = fresh
                self._n_chunks = 0  # everything else comes from `fresh`
                continue
            self._current_chunk_iter = iter(chunk)

    def close(self):
        """Stop iteration. Subsequent ``next()`` raises ``StopIteration``."""
        self._closed = True
        self._current_chunk_iter = None

    def send(self, value):
        raise AttributeError(
            "cached generator: .send() is not supported on chunked "
            "iterators. The cached iterator replays values from the "
            "backend. If you need send() semantics, the function "
            "cannot be cached."
        )

    def throw(self, *args, **kwargs):
        raise AttributeError(
            "cached generator: .throw() is not supported on chunked "
            "iterators. If you need throw() semantics, the function "
            "cannot be cached."
        )


def is_one_shot_iterator(value: Any) -> bool:
    """Return True if *value* is its own iterator (a one-shot consumable).

    Matches Python generators, ``map``/``filter``/``zip`` results, and
    custom iterators that return ``self`` from ``__iter__``. Returns
    False for collections (``list``/``dict``/``set``/``tuple``/``str``/
    ``range``) which are iterable but return fresh iterators on
    ``iter()`` - those are safely cacheable as-is.
    """
    try:
        if isinstance(value, io.IOBase):
            # A file object is its own iterator, so this path claimed it: the
            # caller got a replay iterator with no `read`, `write`, `name` or
            # `fileno`, and the handle was drained to build the chunks (found
            # attacking the decorator before round 26). A handle is not a
            # stream of values to replay -- it is a handle.
            return False
        return iter(value) is value
    except TypeError:
        return False

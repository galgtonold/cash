"""Reading a class's annotations does not change the key of its class.

Reading ``cls.__annotations__`` of a class with none of its own stores an
empty dict on it (3.10-3.13); Python 3.14 caches what it computed in
``__annotations_cache__`` and sets ``__annotate_func__``. The class's code
surface folded those in, so a process that had read them (a dataclass
check, ``typing.get_type_hints``, cash's own analysis) keyed a call
differently from one that had not.
"""

from __future__ import annotations

from cash import Cash
from cash.backends import InMemoryBackend


class _Plain:
    def __init__(self, n: int) -> None:
        self.n = n


def test_reading_a_classs_annotations_does_not_change_its_key():
    """Two `Cash` instances over one backend, as two processes over one
    disk: the first keys the call before the annotations were read."""
    backend, runs = InMemoryBackend(), []

    def total(cache: Cash):
        @cache.cache
        def run(obj):
            runs.append(1)
            return obj.n

        return run

    for name in ("__annotations__", "__annotations_cache__", "__annotate_func__"):
        if name in vars(_Plain):
            delattr(_Plain, name)
    assert total(Cash(backend=backend, register_magic=False))(_Plain(3)) == 3
    _ = _Plain.__annotations__
    assert total(Cash(backend=backend, register_magic=False))(_Plain(3)) == 3
    assert len(runs) == 1

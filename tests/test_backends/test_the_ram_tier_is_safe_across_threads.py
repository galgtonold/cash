"""One RAM tier used from several threads at once stays consistent.

Eviction walks the whole store while other threads insert and read, and a
read checks for a key and then indexes it; without a lock between them a
thread raises out of an ordinary ``set`` or ``get``.
"""

from __future__ import annotations

import sys
import threading

from cash.backends.memory_backend import InMemoryBackend


def _hammer(backend: InMemoryBackend, threads: int = 6, rounds: int = 800) -> list[str]:
    errors: list[str] = []

    def work(t: int) -> None:
        try:
            for i in range(rounds):
                backend.set(f"{t}-{i % 300}", list(range(50)), {})
                backend.get(f"{(t + 1) % threads}-{i % 300}")
                if i % 50 == 0:
                    backend.list_entries()
        except Exception as exc:  # collected for the assertion
            errors.append(repr(exc))

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    try:
        workers = [threading.Thread(target=work, args=(t,)) for t in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
    finally:
        sys.setswitchinterval(old)
    return errors


def test_concurrent_writes_and_reads_under_a_byte_cap_never_raise():
    backend = InMemoryBackend(max_size_bytes=200_000)
    assert _hammer(backend) == []

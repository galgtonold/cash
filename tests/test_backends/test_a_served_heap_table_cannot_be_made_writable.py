"""A big table the RAM tier copied out of glibc's heap is served frozen,
and no holder of a hit can make it writable again.

`heap_blocks.fresh_copy` puts the copy in an ``mmap``. A frozen array whose
chain of bases ends in a writable ``mmap`` can be made writable again
(``setflags(write=True)``), unlike one that owns its memory: the first
write through a hit then changed what every later call got.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from cash.backends import heap_blocks

_CHILD = textwrap.dedent(
    """
    import ctypes
    libc = ctypes.CDLL("libc.so.6")
    libc.mallopt(-3, 32 << 20)  # M_MMAP_THRESHOLD, fixed
    libc.mallopt(-1, 1 << 30)   # M_TRIM_THRESHOLD: freed heap stays heap
    import numpy as np
    import pandas as pd
    holes = [np.ones(2 << 20) for _ in range(12)]  # 192 MB of heap, then free
    del holes
    from cash import Cash
    from cash.backends.memory_backend import InMemoryBackend

    def signal(seed):
        return pd.DataFrame(np.full((5000, 1000), float(seed)))  # 40 MB, in the heap

    f = Cash(backend=InMemoryBackend(), register_magic=False).cache(signal)
    f(1)
    held = f(1).to_numpy()
    try:
        held.setflags(write=True)
        held[0, 0] = -1.0
    except ValueError:
        pass  # read-only for good, as a table that owns its memory
    print(f(1).iloc[0, 0])
    """
)


@pytest.mark.skipif(not heap_blocks._glibc(), reason="glibc's heap only")
def test_a_served_heap_table_cannot_be_made_writable():
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _CHILD], capture_output=True, text=True, timeout=300, check=True
    )
    assert float(out.stdout.strip().splitlines()[-1]) == 1.0

"""A cache directory that cannot be USED must not take the program with it.

Round-16 gate finding (BLOCKING), the sibling of the unwritable-directory hang.
That one was about a directory cash could create and not write to; this is about
one it cannot create at all, and the failure was worse: ``os.makedirs`` raised
straight out of ``get()``, so the caller's program died — **exit 1, no result,
the body never ran** — before doing any of its own work. Measured on three
shapes, 3/3:

* a directory the process may not create (``C:\\Windows\\System32\\...``, a
  read-only volume, a locked-down host)
* a drive or mount that is not there (``Z:\\cashcache``: the share did not mount
  this boot)
* a path that is a FILE (a typo, or someone's leftover)

The tester reached it with no configuration at all: an installed CLI whose cache
follows the working directory, run from a directory the user cannot write to.
"cash cannot cache" became "your job does not run", from a component whose whole
contract is best-effort.

The tier turns itself off instead, says so once, and the process carries on
computing uncached. In a tiered stack the RAM tier is untouched, so an
in-process repeat still hits — which is the control at the bottom of this file.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings

import pytest

from cash import Cash
from cash.backends.file_backend import FileBackend


def _unusable_dirs(tmp_path):
    """The three shapes, as (label, path) pairs, on any platform."""
    a_file = tmp_path / "notadir.txt"
    a_file.write_text("x", encoding="utf-8")
    under_a_file = a_file / "cache"          # a path whose parent is a file
    if sys.platform == "win32":
        missing_volume = "Z:\\cash_missing_volume"
    else:
        missing_volume = "/proc/cash_cannot_create_this"
    return [
        ("path is a file", str(a_file)),
        ("parent is a file", str(under_a_file)),
        ("volume is not there", missing_volume),
    ]


@pytest.mark.parametrize("label", ["path is a file", "parent is a file",
                                   "volume is not there"])
def test_the_backend_degrades_instead_of_raising(tmp_path, label):
    """A miss and a warning, not an exception, whatever is wrong with the path."""
    path = dict(_unusable_dirs(tmp_path))[label]
    backend = FileBackend(cache_dir=path)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert backend.get("k") == (None, None)
        backend.set("k", b"payload", {})
        assert backend.get("k") == (None, None)
        assert backend.list_entries() == []
        backend.delete("k")
        backend.clear()
        backend.shutdown()

    text = "\n".join(str(w.message) for w in rec)
    assert "CACHE-DIR-UNWRITABLE" in text, f"it failed silently:\n{text}"
    assert path in text, "the warning must name the directory"


def test_the_ram_tier_still_caches_when_disk_is_unusable(tmp_path):
    """Degrading is not the same as switching caching off in-process."""
    a_file = tmp_path / "notadir.txt"
    a_file.write_text("x", encoding="utf-8")
    runs: list[int] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Cash(cache_dir=str(a_file), register_magic=False)

        @c.cache(assume_safe=True)
        def work(n):
            runs.append(n)
            return n * 2

        assert work(21) == 42
        assert work(21) == 42

    assert len(runs) == 1, "the RAM tier should still have served the repeat"


_CHILD = """
import sys
import cash

@cash.cache(assume_safe=True)
def work(n):
    print("RAN", file=sys.stderr, flush=True)
    return n * 2

print("RESULT", work(21))
"""


def test_the_process_still_produces_its_answer(tmp_path):
    """The property that actually failed: the user's program runs and exits 0.

    In-process assertions cannot show this — the report was a job that died
    before its own work, and that only exists as an exit code.
    """
    script = tmp_path / "job.py"
    script.write_text(textwrap.dedent(_CHILD), encoding="utf-8")
    a_file = tmp_path / "notadir.txt"
    a_file.write_text("x", encoding="utf-8")

    env = dict(os.environ, CASH_CACHE_DIR=str(a_file))
    proc = subprocess.run([sys.executable, str(script)], env=env, cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=300)

    assert proc.returncode == 0, f"the job died:\n{proc.stderr[-2000:]}"
    assert "RESULT 42" in proc.stdout, proc.stdout
    assert "RAN" in proc.stderr, "the body never ran"


def test_a_usable_directory_is_unaffected(tmp_path):
    """The control: none of the above may cost an ordinary cache its entries."""
    backend = FileBackend(cache_dir=str(tmp_path / "cache"))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        backend.set("k", b"payload", {})
        backend._writes.wait_all()
        assert backend.get("k")[1] == b"payload"
        backend.shutdown()

    assert "CACHE-DIR-UNWRITABLE" not in "\n".join(str(w.message) for w in rec)

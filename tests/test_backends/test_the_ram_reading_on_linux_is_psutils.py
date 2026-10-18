"""The RAM tier's memory reading on Linux is psutil's, without psutil.

The pressure check reads the machine's total memory and the percent in use.
psutil took 16-27 ms to import at a process's first store; on Linux the two
numbers are read from ``/proc/meminfo`` as psutil reads them, and psutil
answers everywhere else and whenever the file says something unexpected.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from cash.backends import memory_backend

linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc/meminfo is Linux's")

#: A real file's shape, with the lines around the two that are read.
MEMINFO = """\
MemTotal:       16384000 kB
MemFree:         1200000 kB
MemAvailable:    {available} kB
Buffers:          300000 kB
Cached:          5000000 kB
SwapCached:            0 kB
Active:          7000000 kB
Inactive:        4000000 kB
SwapTotal:       2097148 kB
SwapFree:        2097148 kB
Shmem:            250000 kB
SReclaimable:     600000 kB
"""


def _psutil_on(tmp_path, monkeypatch, text):
    """psutil's reading of a /proc whose meminfo is *text*, and cash's."""
    import psutil

    (tmp_path / "meminfo").write_text(text, encoding="utf-8")
    monkeypatch.setattr(psutil, "PROCFS_PATH", str(tmp_path))
    real_read = memory_backend.read_code_file
    monkeypatch.setattr(
        memory_backend,
        "read_code_file",
        lambda path: real_read(str(tmp_path / "meminfo")) if path == "/proc/meminfo" else real_read(path),
    )
    return psutil.virtual_memory(), memory_backend._meminfo_reading()


@linux
@pytest.mark.parametrize("available", [9_000_000, 16_384_000, 1, 123_457, 16_383_999])
def test_the_reading_is_psutils_for_the_same_file(tmp_path, monkeypatch, available):
    theirs, ours = _psutil_on(tmp_path, monkeypatch, MEMINFO.format(available=available))
    assert ours is not None
    assert (ours.total, ours.percent) == (theirs.total, theirs.percent)


@linux
def test_a_file_without_mem_available_leaves_it_to_psutil(tmp_path, monkeypatch):
    text = "".join(line + "\n" for line in MEMINFO.splitlines() if not line.startswith("MemAvailable"))
    _, ours = _psutil_on(tmp_path, monkeypatch, text)
    assert ours is None


@linux
@pytest.mark.parametrize(
    "text",
    [
        MEMINFO.format(available=20_000_000),  # more available than there is
        MEMINFO.format(available=0),  # a kernel bug psutil works around
        MEMINFO.format(available=1).replace("16384000 kB", "0 kB", 1),
        MEMINFO.format(available=1).replace("16384000 kB", "16384000 MB", 1),
        "MemTotal: lots\nMemAvailable: some\n",
    ],
)
def test_a_reading_that_does_not_add_up_leaves_it_to_psutil(tmp_path, monkeypatch, text):
    (tmp_path / "meminfo").write_text(text, encoding="utf-8")
    real_read = memory_backend.read_code_file
    monkeypatch.setattr(memory_backend, "read_code_file", lambda path: real_read(str(tmp_path / "meminfo")))
    assert memory_backend._meminfo_reading() is None


@linux
def test_the_live_reading_agrees_with_psutil():
    import psutil

    theirs = psutil.virtual_memory()
    ours = memory_backend._virtual_memory()
    assert ours.total == theirs.total
    assert ours.percent == pytest.approx(theirs.percent, abs=5.0)  # a moment apart on a live machine


def test_elsewhere_psutil_answers(monkeypatch):
    monkeypatch.setattr(memory_backend.sys, "platform", "darwin")
    assert memory_backend._meminfo_reading() is None


@linux
def test_a_store_that_checks_memory_does_not_import_psutil():
    script = textwrap.dedent(
        """
        import sys
        from cash.backends.memory_backend import InMemoryBackend
        b = InMemoryBackend(check_interval=1)
        for i in range(3):
            b.set(f"k{i}", i, {"execution_time": 1.0})
        print("PSUTIL", "psutil" in sys.modules)
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout
    assert "PSUTIL False" in out

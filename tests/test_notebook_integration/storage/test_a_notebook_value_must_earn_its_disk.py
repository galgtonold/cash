"""The bytes ceiling reaches the path notebook values actually take.

The unit arm is ``tests/test_backends/test_a_value_must_earn_its_disk.py``.
This file exists because passing that arm did not mean the rule was running
where it matters.

A notebook statement sets ``defer_persist``, so ``TieredBackend.set`` skips its
whole promotion block for it and the value reaches disk later, from the
end-of-cell pass (``persist_from_memory``). A ceiling placed only in ``set``
therefore governs the decorator and direct callers -- and misses every notebook
statement, which is the entire population that filled the measured caches:
9.7 GB of loop iterations in one and 15.3 GB of frames in another are notebook
statements.
It looked like it worked, because the pure predicate was right and the
backend-level tests passed.

Two arms, on the real notebook path:

  * a large value that is cheap to rebuild does not reach disk
  * a large value that is expensive to rebuild still does (the control;
    ``test_large_frame_reaches_disk.py`` is the standing regression test for refusing
    those, and it must keep passing)
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]


def _disk_bytes(nb_runner, cell: int) -> int:
    report = nb_runner.get_output(cell)
    assert "MAX_DATA_BYTES=" in report, report
    return int(report.split("MAX_DATA_BYTES=")[1].split()[0])


REPORT = (
    "import os, glob\n"
    "_cash = os.path.abspath('.cash')\n"
    "_datas = glob.glob(os.path.join(_cash, '*.entry'))\n"
    "_maxb = max((os.path.getsize(f) for f in _datas), default=0)\n"
    "print(f'DATA_FILES={len(_datas)} MAX_DATA_BYTES={_maxb}')"
)


def test_a_big_value_that_rebuilds_quickly_does_not_reach_disk(nb_runner):
    """~96 MB built in about 0.17 s: over 500 MiB per compute-second.

    The reported shape -- 48 MiB entries whose recorded compute was 0.00 s, 72 of
    them, 3.5 GB. Nothing this cheap is worth that much disk.

    The sleep keeps the compute above the 0.1 s persistence floor, and the
    refusal must name the bytes ceiling, so nothing else can be what keeps it
    off disk. The value is twice the reported size so the ceiling (128 MiB per
    compute-second) is crossed only past ~0.72 s of recorded compute, about
    four times the ~0.17 s it takes. At 48 MB it was crossed at ~0.36 s, which
    a loaded machine can reach, and then the value is written for a
    legitimate reason.
    """
    nb_runner.create_notebook(
        [
            "import numpy as np\nimport time",
            "def make_frame():\n    time.sleep(0.15)\n    return np.arange(12_000_000, dtype='float64')",
            "big = make_frame()\nprint(f'shape={big.shape[0]}')",
            REPORT,
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert "shape=12000000" in nb_runner.get_output(3)
    raw = nb_runner.get_raw_output(3)
    assert _disk_bytes(nb_runner, 4) < 1_000_000, (
        f"a 96 MB value that rebuilds in 0.15 s was written to disk anyway:\n{raw}"
    )
    assert "CACHE-NOT-WORTH-BYTES" in raw, f"kept off disk, but not by the bytes ceiling:\n{raw}"


def test_a_big_value_that_is_expensive_still_reaches_disk(nb_runner):
    """The control. The ceiling must not simply refuse everything large --
    that was a bug once, and re-creating it would be worse than the
    disk it saves. The same value as the arm above; only the compute differs."""
    nb_runner.create_notebook(
        [
            "import numpy as np\nimport time",
            "def make_frame():\n    time.sleep(2.0)\n    return np.arange(12_000_000, dtype='float64')",
            "big = make_frame()\nprint(f'shape={big.shape[0]}')",
            REPORT,
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert "shape=12000000" in nb_runner.get_output(3)
    assert _disk_bytes(nb_runner, 4) > 1_000_000, "a 96 MB value that costs 2 s to rebuild should be on disk"

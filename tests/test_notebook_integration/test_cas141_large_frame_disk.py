"""Integration test for CAS-141: large, expensive results reach the disk tier.

Before the fix, the tiered backend's promotion policy modelled raw byte-I/O at a
hardcoded 100 MB/s with zero serialization cost, so the ``io_time`` grew with
size while compute did not — the policy *inverted* and left big frames RAM-only
(the reported symptom: a ``.cash`` holding 393 KB for a 0.68 GB workload). The
serialization-aware cost model now promotes them.

This exercises the real notebook path end to end: a cell produces a ~48 MB array
that took long enough to compute to clear the floor, and we assert a real value
blob (a ``*.data`` file, not a metadata-only ``*.meta`` stub) landed on disk.
"""
import glob
import os

import pytest


pytestmark = [pytest.mark.restore]


class TestLargeFrameReachesDisk:
    def test_expensive_large_array_writes_value_to_disk(self, nb_runner):
        """A ~48 MB array from an expensive statement is promoted to disk.

        Under the old inverted policy a 48 MB result computed in < 1 s was
        refused (``execution_time < smart_persistence_threshold``) and never
        left RAM, so no ``*.data`` file was written.
        """
        nb_runner.create_notebook([
            "import numpy as np\nimport time",
            # One cached statement whose compute (the sleep) clears the 0.1 s
            # floor and whose output is large (~48 MB, incompressible enough
            # that compress=False keeps it big on disk).
            "def make_frame():\n"
            "    time.sleep(0.6)\n"
            "    return np.arange(6_000_000, dtype='float64')",
            "big = make_frame()\nprint(f'shape={big.shape[0]} sum_ok={big[-1]}')",
            # Report the on-disk cache contents from the kernel's own cwd, so
            # the assertion is independent of where the work dir lives.
            "import os, glob\n"
            "_cash = os.path.abspath('.cash')\n"
            "_datas = glob.glob(os.path.join(_cash, '*.data'))\n"
            "_maxb = max((os.path.getsize(f) for f in _datas), default=0)\n"
            "print(f'DATA_FILES={len(_datas)} MAX_DATA_BYTES={_maxb}')",
        ])
        nb_runner.start_kernel()
        nb_runner.run_all()

        assert "shape=6000000" in nb_runner.get_output(3)

        report = nb_runner.get_output(4)
        assert "MAX_DATA_BYTES=" in report, report
        max_bytes = int(report.split("MAX_DATA_BYTES=")[1].split()[0])
        # The 48 MB value blob must be on disk. A metadata-only entry is a few
        # hundred bytes, so a multi-MB *.data file proves the value persisted.
        assert max_bytes > 1_000_000, (
            f"expected a large *.data blob on disk, got {max_bytes} bytes: {report}"
        )

        # And it restores correctly after cash's tracking state is reset.
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "shape=6000000" in nb_runner.get_output(3)

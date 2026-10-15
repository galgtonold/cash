"""Integration test: large, expensive results reach the disk tier.

Before the fix, the tiered backend's promotion policy modelled raw byte-I/O at a
hardcoded 100 MB/s with zero serialization cost, so the ``io_time`` grew with
size while compute did not — the policy *inverted* and left big frames RAM-only
(the reported symptom: a ``.cash`` holding 393 KB for a 0.68 GB workload). The
serialization-aware cost model now promotes them.

This exercises the real notebook path end to end: a cell produces a ~48 MB array
that took long enough to compute to clear the floor, and we assert a real value
blob landed on disk, rather than a metadata-only stub carrying no value.
"""

import pytest

from tests._nbharness.badge import shows_cached
from tests._nbharness.disk import ENTRY_BYTES_REPORT

pytestmark = [pytest.mark.restore]


class TestLargeFrameReachesDisk:
    def test_expensive_large_array_writes_value_to_disk(self, nb_runner):
        """A ~48 MB array from an expensive statement is promoted to disk.

        Under the old inverted policy a 48 MB result computed in < 1 s was
        refused (its compute fell under the promotion floor) and never left
        RAM, so the entry on disk held no payload.
        """
        nb_runner.create_notebook(
            [
                "import cash\n%load_ext cash\n%cash_badge print\n%cash_on\nimport numpy as np\nimport time",
                # One cached statement whose compute (the sleep) clears the 0.1 s
                # floor and whose output is large (~48 MB, incompressible enough
                # that compress=False keeps it big on disk).
                "def make_frame():\n    time.sleep(0.6)\n    return np.arange(6_000_000, dtype='float64')",
                "big = make_frame()\nprint(f'shape={big.shape[0]} sum_ok={big[-1]}')",
                # Report the on-disk cache contents from the kernel's own cwd, so
                # the assertion is independent of where the work dir lives. An
                # entry counts with the chunks it holds: the array's data is in
                # the chunk store, not in the ``.entry`` file.
                ENTRY_BYTES_REPORT,
            ]
        )
        nb_runner.start_kernel()
        # The report reads the folder after the writes land: a cell does not
        # wait for them, and on a loaded machine they may still be queued.
        nb_runner.run_cells([1, 2, 3])
        nb_runner.settle_writes()
        nb_runner.run_cell(4)

        assert "shape=6000000" in nb_runner.get_output(3)

        report = nb_runner.get_output(4)
        assert "MAX_DATA_BYTES=" in report, report
        max_bytes = int(report.split("MAX_DATA_BYTES=")[1].split()[0])
        # The 48 MB value must be on disk. A metadata-only entry is a few
        # hundred bytes, so a multi-MB entry (with its chunks) proves the
        # value persisted.
        assert max_bytes > 1_000_000, f"expected a large entry on disk, got {max_bytes} bytes: {report}"

        # And it restores correctly after cash's tracking state is reset.
        nb_runner.reset_cash_state()
        nb_runner.run_all()
        assert "shape=6000000" in nb_runner.get_output(3)

        # And from disk alone: a new kernel has nothing in RAM, so the
        # statement comes back from the entry and its chunks or runs again.
        nb_runner.restart()
        nb_runner.run_all()
        out = nb_runner.get_raw_output(3)
        assert "shape=6000000" in out, out
        assert shows_cached(out), f"the 48 MB value was not restored from disk after a restart:\n{out}"

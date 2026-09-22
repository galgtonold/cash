"""A timezone file read once per process does not change a load's lineage.

Round 30, r30s1 (2/2): after editing a setup constant and re-running, the
first restart re-ran the whole chain below the load. The load's lineage
carried the files it read, and on the first load in a process that included
``tzdata/zoneinfo/UTC`` -- ``zoneinfo`` reads a zone once and keeps it for the
life of the process. The in-session re-run did not read it, the run after the
restart did, so the slow step's key moved and it missed.

The reader is the standard library (``zoneinfo``) and the file belongs to
another installed package (``tzdata``), so "a library reading its own package"
did not cover it.
"""
import pytest

pytest.importorskip("tzdata")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]


def _cells(folder, n):
    return [
        "import cash\n%cash_on\n%cash_badge print\nimport time, zoneinfo, glob",
        f"N_DAYS = {n}",
        f"files = sorted(glob.glob(r'{folder}/day_*.txt'))[:N_DAYS]",
        "def load_days(fs):\n"
        "    tz = zoneinfo.ZoneInfo('Europe/Vienna')\n"
        "    return [(open(f).read(), str(tz)) for f in fs]\n"
        "raw = load_days(files)",
        "def slow(r):\n    time.sleep(1.5)\n    return len(r)\n"
        "out = slow(raw)",
        "print('OUT', out)",
    ]


def test_the_slow_step_restores_after_an_edit_and_a_restart(nb_runner, tmp_path):
    folder = tmp_path / "days"
    folder.mkdir()
    for i in range(5):
        (folder / f"day_{i}.txt").write_text(f"day {i}\n")
    folder = folder.as_posix()

    nb_runner.create_notebook(_cells(folder, 2))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "OUT 2" in nb_runner.get_output(6), nb_runner.get_raw_output(6)

    nb_runner.set_cell_source(2, "N_DAYS = 4")
    for i in range(2, 7):
        nb_runner.run_cell(i)
    assert "OUT 4" in nb_runner.get_output(6), nb_runner.get_raw_output(6)

    nb_runner.restart()
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(5)
    assert "OUT 4" in nb_runner.get_output(6), nb_runner.get_raw_output(6)
    assert "EXECUTED: out = slow" not in raw, (
        "the first restart after the edit re-ran the slow step:\n" + raw)

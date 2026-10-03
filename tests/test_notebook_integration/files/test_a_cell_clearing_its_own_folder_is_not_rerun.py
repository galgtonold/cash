"""A cell that clears its output folder and redraws it is left alone by the
cells below it that do not read the folder.

Reproduced 3/3: a chart cell started with ``for old in CHARTS.glob('*.png'):
old.unlink()`` and then saved a chart per feeder. Every later cell, even
``x = 41 + 1``, re-ran the deletion and the redraw first. The loop opens no
file, so its run left no record that it had run; what it deletes is named by
the listing, so it could not be ruled out as unread; and ``os.remove(f)``
was read as ``list.remove`` changing ``os``, so any cell using ``os`` asked
for the cell again.

A cell that lists the folder still sees what changes it: a file put there
from outside, or an edit upstream of the cell that writes it.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(120)]

CLEARS = {
    "path_glob": "for old in OUT.glob('*.txt'):\n    old.unlink()",
    "glob_glob": "for f in glob.glob(str(OUT / '*.txt')):\n    os.remove(f)",
    "listdir": "for name in os.listdir(OUT):\n    os.remove(os.path.join(OUT, name))",
}


def _chart_cell(out, clear, n="3"):
    return (
        f"import glob, os, sys\nfrom pathlib import Path\nOUT = Path(r'{out}')\nOUT.mkdir(exist_ok=True)\n"
        + CLEARS[clear]
        + f"\nfor k in range({n}):\n    print('RUN draw', k, file=sys.stderr)\n"
        "    (OUT / f'c{k}.txt').write_text(str(k))"
    )


@pytest.mark.parametrize("unrelated", ["x = 41 + 1", "x = os.path.basename('a/b')"], ids=["plain", "uses_os"])
@pytest.mark.parametrize("clear", sorted(CLEARS))
def test_an_unrelated_cell_does_not_rerun_it(nb_runner, tmp_path, clear, unrelated):
    out = tmp_path / "charts"
    out.mkdir()
    (out / "from_last_session.txt").write_text("old", encoding="utf-8")
    nb_runner.create_notebook(
        ["import cash\n%cash_on", _chart_cell(out, clear), unrelated + "\nprint('RUN unrelated', x)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert sorted(p.name for p in out.iterdir()) == ["c0.txt", "c1.txt", "c2.txt"]

    # Not the cell's file: a re-run of the chart cell deletes it.
    (out / "kept.txt").write_text("kept", encoding="utf-8")
    for _ in range(3):
        nb_runner.run_cell(3)
        assert "RUN unrelated" in nb_runner.get_output(3)
        assert "RUN draw" not in nb_runner.get_output(3), nb_runner.get_raw_output(3)
    assert (out / "kept.txt").exists()


@pytest.mark.fresh_kernel
@pytest.mark.parametrize("clear", sorted(CLEARS))
def test_an_unrelated_cell_does_not_rerun_it_after_a_restart(nb_runner, tmp_path, clear):
    out = tmp_path / "charts"
    nb_runner.create_notebook(
        ["import cash\n%cash_on", _chart_cell(out, clear), "x = 41 + 1\nprint('RUN unrelated', x)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.restart()
    (out / "kept.txt").write_text("kept", encoding="utf-8")
    nb_runner.run_cell(1)
    nb_runner.run_cell(3)
    assert "RUN unrelated" in nb_runner.get_output(3)
    assert "RUN draw" not in nb_runner.get_output(3), nb_runner.get_raw_output(3)
    assert (out / "kept.txt").exists()


@pytest.mark.parametrize("clear", sorted(CLEARS))
def test_a_cell_listing_the_folder_sees_what_else_changes_it(nb_runner, tmp_path, clear):
    out = tmp_path / "charts"
    cells = [
        "import cash\n%cash_on",
        "N = 2",
        _chart_cell(out, clear, n="N"),
        "names = sorted(os.listdir(OUT))\nprint('NAMES', names)",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "NAMES ['c0.txt', 'c1.txt']" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    for _ in range(2):
        nb_runner.run_cell(4)
        assert "NAMES ['c0.txt', 'c1.txt']" in nb_runner.get_output(4), nb_runner.get_raw_output(4)
        assert "RUN draw" not in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    # Something else changes the folder: the listing is read again, and the
    # chart cell, which did not change, is not re-run (it would delete the file).
    (out / "added.txt").write_text("a", encoding="utf-8")
    nb_runner.run_cell(4)
    assert "NAMES ['added.txt', 'c0.txt', 'c1.txt']" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    # An edit above the chart cell: it is re-run for the cell that lists its folder.
    nb_runner.set_cell_source(2, "N = 3")
    nb_runner.run_cell(4)
    assert "'c2.txt'" in nb_runner.get_output(4), nb_runner.get_raw_output(4)


@pytest.mark.parametrize("dropped", [False, True], ids=["fewer_charts", "file_dropped_in"])
@pytest.mark.parametrize("clear", sorted(CLEARS))
def test_a_replayed_redraw_clears_the_folder_first(nb_runner, tmp_path, clear, dropped):
    """An edit above the chart cell replays its drawing for the cell listing
    the folder; the deletion that clears the folder replays with it. Without
    it, ``N`` going from 3 to 2 left ``c2.txt`` behind, and a file dropped in
    from outside survived a replay that a run of the cell removes."""
    out = tmp_path / "charts"
    cells = [
        "import cash\n%cash_on",
        "N = 3",
        _chart_cell(out, clear, n="N"),
        "names = sorted(os.listdir(OUT))\nprint('NAMES', names)",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "NAMES ['c0.txt', 'c1.txt', 'c2.txt']" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    if dropped:
        (out / "added.txt").write_text("a", encoding="utf-8")
        nb_runner.set_cell_source(2, "N = 4")
        want = ["c0.txt", "c1.txt", "c2.txt", "c3.txt"]
    else:
        nb_runner.set_cell_source(2, "N = 2")
        want = ["c0.txt", "c1.txt"]
    nb_runner.run_cell(4)
    assert f"NAMES {want!r}" in nb_runner.get_output(4), nb_runner.get_raw_output(4)
    assert sorted(p.name for p in out.iterdir()) == want

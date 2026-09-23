"""A cell reading files through a list of paths leaves an unrelated chart alone.

Reproduced 3/3: after an upstream edit a chart above was out
of date, and the badge said so with a STALE FILE line -- as documented, a
writer whose file the cell you run does not read is left alone. Then a new
cell below it, ``TF = [Path('other.csv')]`` and
``pd.concat([pd.read_csv(f) for f in TF])``, re-drew the chart. It reads
neither the chart nor anything drawn from it; its read went through a list, so
what it read was "unknown", and an unknown read cannot rule the chart out.
"""

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(300)]


@pytest.mark.parametrize("spelling", ["same_cell", "earlier_cell"])
def test_the_chart_is_not_redrawn(nb_runner, tmp_path, spelling):
    folder = tmp_path.as_posix()
    import pandas as pd

    pd.DataFrame({"g": [i % 5 for i in range(200)], "v": range(200)}).to_csv(tmp_path / "data.csv", index=False)
    pd.DataFrame({"g": range(5), "t": range(5)}).to_csv(tmp_path / "other.csv", index=False)
    chart = tmp_path / "report" / "chart.png"

    read_cell = (f"TF = [Path(r'{folder}/other.csv')]\n" if spelling == "same_cell" else "") + (
        "t = pd.concat([pd.read_csv(f) for f in TF])\n"
        "attain = total.to_frame('m').join(t.set_index('g'))\nprint('ROWS', len(attain))"
    )
    cells = [
        "import cash\n%cash_on\n%cash_badge print\nimport pandas as pd\nfrom pathlib import Path\n"
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt",
        f"REPORT = Path(r'{folder}/report'); REPORT.mkdir(exist_ok=True)\nK = 1.0"
        + (f"\nTF = [Path(r'{folder}/other.csv')]" if spelling == "earlier_cell" else ""),
        f"data = pd.read_csv(r'{folder}/data.csv')\ndata = data.assign(v=data['v'] * K)",
        "weekly = data.groupby('g')['v'].sum()",
        "fig, ax = plt.subplots()\nweekly.plot(ax=ax)\nfig.savefig(REPORT / 'chart.png')\nplt.close(fig)",
        "total = data.groupby('g')['v'].mean()\nprint(total.sum())",
        read_cell,
    ]
    nb_runner.create_notebook(cells[:6])
    nb_runner.start_kernel()
    nb_runner.run_all()
    drawn = chart.stat().st_mtime_ns

    nb_runner.set_cell_source(2, cells[1].replace("K = 1.0", "K = 2.0"))
    nb_runner.run_cell(6)
    assert "STALE FILE" in nb_runner.get_raw_output(6), nb_runner.get_raw_output(6)
    assert chart.stat().st_mtime_ns == drawn

    nb_runner.add_cell(read_cell, save=True)
    nb_runner.run_cell(7)
    assert "ROWS 5" in nb_runner.get_output(7), nb_runner.get_raw_output(7)
    assert chart.stat().st_mtime_ns == drawn, (
        "a cell that reads other.csv re-drew the chart:\n" + nb_runner.get_raw_output(7)
    )

"""A repair does not run past a cell a run from the top cannot get through.

Cell 5 fills ``nps`` from ``sv``, which only cell 6 below it binds: run from
the top, it raises ``NameError`` and nothing below it runs. It raised once
under cash, then ran on the ``sv`` cell 6 had left in memory, and the
report in cell 8 wrote ``nps`` to a file. After the data changed and cell 2
was run again, running the last cell made cash rebuild what had moved: it
re-ran the report, and with it the ``nps = {}`` of cell 5 without the loop
that fills it, which its simulation left out. ``nps`` came back empty and
the report wrote the empty dict to its file.

No run rebuilds what the kernel got through cell 5 with, so cash re-runs
nothing at or below it: the cell runs on the values in memory, uncached,
with NOTEBOOK-REPAIR-GAP.
"""

import json

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(120)]

CELLS = [
    "import cash\n%cash_on",
    "import json\nfrom pathlib import Path\nimport pandas as pd\nOUT = Path('out')\nOUT.mkdir(exist_ok=True)",
    "src = pd.read_csv('data.csv')\ndf = src.assign(y=src['x'] * 2)",
    "def seg(frame):\n    frame['seen'] = 1\n    return frame.groupby('k')['y'].sum()",
    "by = seg(df)",
    "nps = {}\nfor v, g in sv.groupby('k'):\n    nps[v] = int(g['x'].sum())\nnps",
    "sv = src.copy()",
    "with open(OUT / 'by.txt', 'w') as f:\n    f.write(str(by.to_dict()))",
    "summary = {'nps': nps, 'n': len(df)}\nwith open(OUT / 'summary.json', 'w') as f:\n    json.dump(summary, f)",
    "daily = df.groupby('k').size()",
]


def _run(nb_runner, n):
    try:
        nb_runner.run_cell(n)
    except Exception as e:  # a cell that raises, as the user ran it
        return str(e)
    return ""


def test_the_filled_dict_and_its_report_are_kept(nb_runner):
    work = nb_runner.work_dir
    (work / "data.csv").write_text("k,x\n1,1\n1,2\n2,3\n", encoding="utf-8")
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    for n in [2, 3, 4, 5]:
        _run(nb_runner, n)
    assert _run(nb_runner, 6), "cell 6 must raise: only cell 7 binds sv"
    for n in [7, 6, 8, 9]:
        _run(nb_runner, n)
    assert nb_runner.peek("repr(globals()['nps'])") == "'{1: 3, 2: 3}'"
    report = (work / "out" / "summary.json").read_text(encoding="utf-8")

    (work / "data.csv").write_text("k,x\n1,1\n1,2\n2,3\n2,7\n", encoding="utf-8")
    _run(nb_runner, 3)
    _run(nb_runner, 10)

    assert nb_runner.peek("repr(globals()['nps'])") == "'{1: 3, 2: 3}'"
    assert (work / "out" / "summary.json").read_text(encoding="utf-8") == report
    assert json.loads(report)["nps"] == {"1": 3, "2": 3}
    assert "NOTEBOOK-REPAIR-GAP" in nb_runner.get_raw_output(10)
    assert nb_runner.peek("globals()['daily'].tolist()") == "[2, 2]"

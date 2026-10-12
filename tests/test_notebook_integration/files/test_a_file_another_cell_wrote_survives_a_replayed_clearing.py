"""A file another cell wrote survives a folder-clearing cell replayed above it.

An export cell clears ``export/`` and writes one CSV per region; the next
cell writes ``summary.csv`` into the same folder; a manifest cell lists it.
After the region list was edited and only the manifest cell run, the
repair replayed the clearing cell, which removed ``summary.csv``, and not
the summary writer: the listing and the disk lost it, with no warning.
"""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]

CELLS = [
    "import os, glob\nimport pandas as pd\nREGIONS = ['north', 'south']\nOUT = 'export'\n"
    "os.makedirs(OUT, exist_ok=True)\n"
    "df = pd.DataFrame({'region': ['north', 'south', 'east'] * 4, 'amount': range(12)})",
    "for old in glob.glob(os.path.join(OUT, '*.csv')):\n    os.remove(old)\n"
    "for reg in REGIONS:\n    df[df.region == reg].to_csv(os.path.join(OUT, f'{reg}.csv'), index=False)",
    "df.groupby('region').amount.sum().to_csv(os.path.join(OUT, 'summary.csv'))",
    "manifest = sorted(os.listdir(OUT))\nprint(manifest)",
]


def test_running_only_the_listing_keeps_the_summary(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(4).strip() == "['north.csv', 'south.csv', 'summary.csv']"
    nb_runner.set_cell_source(1, CELLS[0].replace("['north', 'south']", "['north', 'east']"))
    nb_runner.run_cell(4)
    assert nb_runner.peek("manifest") == "['east.csv', 'north.csv', 'summary.csv']"
    assert sorted(p.name for p in (Path(nb_runner.work_dir) / "export").iterdir()) == [
        "east.csv",
        "north.csv",
        "summary.csv",
    ]

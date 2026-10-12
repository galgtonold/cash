"""A writer whose file a re-run statement above it removes runs again.

An export cell clears its folder and writes one CSV per region; another cell
writes ``summary.csv`` into the same folder; a manifest cell lists it. After
the region list was edited and only the manifest cell run, the upstream
repair re-ran the clearing cell -- which removed ``summary.csv`` -- but not
the summary writer, judged fresh from the disk as it was before the repair.
The manifest, and the folder, silently lost the summary.
"""

from __future__ import annotations

import os

from tests._cell_driver import run_cash_cell

SETUP = (
    "import os, glob\nimport pandas as pd\nREGIONS = ['north', 'south']\nOUT = 'export'\n"
    "os.makedirs(OUT, exist_ok=True)\nos.makedirs('elsewhere', exist_ok=True)\n"
    "df = pd.DataFrame({'region': ['north', 'south', 'east'] * 4, 'amount': range(12)})"
)
EXPORT = (
    "for old in glob.glob(os.path.join(OUT, '*.csv')):\n    os.remove(old)\n"
    "for reg in REGIONS:\n    df[df.region == reg].to_csv(os.path.join(OUT, f'{reg}.csv'), index=False)"
)
MANIFEST = "manifest = sorted(os.listdir(OUT))"


def _cells(summary_folder: str) -> list[str]:
    return [
        SETUP,
        EXPORT,
        f"df.groupby('region').amount.sum().to_csv(os.path.join({summary_folder}, 'summary.csv'))",
        MANIFEST,
    ]


def _edit_regions_and_run_the_manifest(magics, cells):
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)
    cells[0] = cells[0].replace("['north', 'south']", "['north', 'east']")
    run_cash_cell(magics, cells[3], cells=cells)


def test_the_summary_is_written_again(cash_magics, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    _edit_regions_and_run_the_manifest(cash_magics, _cells("OUT"))
    assert cash_magics.shell.user_ns["manifest"] == ["east.csv", "north.csv", "summary.csv"]
    assert sorted(os.listdir(tmp_path / "export")) == ["east.csv", "north.csv", "summary.csv"]


def test_a_writer_into_another_folder_is_left_alone(cash_magics, tmp_path, monkeypatch):
    """Control: nothing the repair runs touches its file."""
    monkeypatch.chdir(tmp_path)
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    cells = _cells("'elsewhere'")
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    summary = tmp_path / "elsewhere" / "summary.csv"
    os.utime(summary, ns=(0, 0))
    cells[0] = cells[0].replace("['north', 'south']", "['north', 'east']")
    run_cash_cell(cash_magics, cells[3], cells=cells)
    assert cash_magics.shell.user_ns["manifest"] == ["east.csv", "north.csv"]
    assert os.stat(summary).st_mtime_ns == 0, "the summary was written again"

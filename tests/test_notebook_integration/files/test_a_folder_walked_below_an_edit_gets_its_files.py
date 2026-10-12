"""A cell listing an output folder with ``os.walk`` gets the files the cells
above write into it, after an edit above and a run of that cell alone.

Only ``os.listdir`` / ``Path.iterdir`` / ``glob`` counted as reading the
folder: the export cell was not run again for the walk, and the listing
showed the files of the old region list.
"""

import pytest

pytestmark = [pytest.mark.integration]

SETUP = (
    "import os, glob\nimport pandas as pd\nREGIONS = ['north', 'south']\nOUT = 'export'\n"
    "os.makedirs(OUT, exist_ok=True)\n"
    "df = pd.DataFrame({'region': ['north', 'south', 'east'] * 4, 'amount': range(12)})\n"
)
EXPORT = (
    "for old in glob.glob(os.path.join(OUT, '*.csv')):\n    os.remove(old)\n"
    "for reg in REGIONS:\n    df[df.region == reg].to_csv(os.path.join(OUT, f'{reg}.csv'), index=False)"
)
MANIFEST = "manifest = sorted(f for _, _, fs in os.walk(OUT) for f in fs)\nprint(manifest)"


def test_the_walk_sees_the_new_files(nb_runner):
    nb_runner.create_notebook([SETUP, EXPORT, MANIFEST])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip().endswith("['north.csv', 'south.csv']")
    nb_runner.set_cell_source(1, SETUP.replace("['north', 'south']", "['north', 'east']"))
    nb_runner.run_cell(3)
    assert nb_runner.get_output(3).strip().endswith("['east.csv', 'north.csv']")

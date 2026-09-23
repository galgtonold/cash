"""A statement that looks at a file's size or timestamps depends on that file.

Round 24's r24s4 ended its export cell with
``print({p.name: p.stat().st_size for p in sorted(OUT.glob("*.csv"))})``.
When the exports above it were rewritten, that line was served from the cache
and showed the sample run's sizes: the folder's listing is a dependency, and
the names had not changed; what ``stat`` reports was not a dependency at all.
Exports on disk were right; the numbers on screen were not.
"""

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.files]

SETUP = (
    "import os\nimport pandas as pd\nfrom pathlib import Path\nOUT = Path('out_r')\nOUT.mkdir(exist_ok=True)\nN = {n}"
)


@pytest.mark.parametrize(
    "shown",
    [
        "print('SIZES', {p.name: p.stat().st_size for p in sorted(OUT.glob('*.csv'))})",
        "print('SIZES', {'a.csv': os.path.getsize(OUT / 'a.csv')})",
    ],
    ids=["path_stat", "os_path_getsize"],
)
def test_a_size_shown_after_the_file_changed_is_the_new_one(nb_runner, shown):
    # `lineterminator` pinned: `to_csv` defaults to the platform's line ending,
    # and the sizes below would differ between Windows and everywhere else.
    export = "pd.DataFrame({'x': range(N)}).to_csv(OUT / 'a.csv', index=False, lineterminator='\\n')\n" + shown
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP.format(n=10), export])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "SIZES {'a.csv': 22}" in nb_runner.get_output(3), nb_runner.get_output(3)

    nb_runner.set_cell_source(2, SETUP.format(n=1000))
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)

    assert "SIZES {'a.csv': 3892}" in nb_runner.get_output(3), nb_runner.get_output(3)

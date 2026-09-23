"""A file replaced in place is seen, even after a new file arrived before it.

Round 23 (r23s2, 4/4, also on Restart & Run All). A folder of CSVs is read in
a loop into ``raw``; the next cell derives ``sales``. Session 2: a new file
lands. Session 3: an existing file is rewritten in place. ``raw`` had the new
rows, but ``sales`` was restored with session 2's entry -- the old content.
Without the new file in between, the rewrite was caught.
"""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]

FILES = 200
ROWS = 6000

READ = (
    "import glob, os\nimport pandas as pd\n"
    "files = sorted(glob.glob(os.path.join('exports', 'pos_*.csv')))\n"
    "parts = []\n"
    "for f in files:\n"
    "    d = pd.read_csv(f)\n"
    "    d['source_file'] = os.path.basename(f)\n"
    "    parts.append(d)\n"
    "raw = pd.concat(parts, ignore_index=True)\n"
    "print(len(files), 'files', len(raw), 'rows')"
)
DERIVE = (
    "key_cols = [c for c in raw.columns if c != 'source_file']\n"
    "sales = raw.drop_duplicates(subset=key_cols).copy()\n"
    "print('S030 total:', int(sales.loc[sales['store'] == 'S030', 'v'].sum()))"
)


def _frame(store, n, rng):
    return pd.DataFrame({"store": store, "receipt": [f"R{store}-{k}" for k in range(n)], "v": rng.integers(1, 100, n)})


def _session(nb_runner):
    nb_runner.restart()
    nb_runner.run_all()
    return nb_runner.get_output(3)


@pytest.mark.parametrize("new_file_first", [True, False], ids=["new-file-first", "rewrite-only"])
def test_the_rewritten_file_reaches_the_derived_frame(nb_runner, new_file_first):
    rng = np.random.default_rng(1)
    folder = Path(nb_runner.work_dir) / "exports"
    folder.mkdir()
    for i in range(FILES):
        _frame(f"S{i:03d}", ROWS, rng).to_csv(folder / f"pos_S{i:03d}.csv", index=False)
    nb_runner.create_notebook(["import cash\n%cash_on", READ, DERIVE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    if new_file_first:
        _frame("S999", ROWS, rng).to_csv(folder / "pos_S999.csv", index=False)
        _session(nb_runner)
    target = folder / "pos_S030.csv"
    df = pd.read_csv(target)
    df.loc[df.index % 25 == 0, "v"] *= 2  # 4% of the values doubled
    df = df[df.index % 100 != 7]  # 1% of the rows dropped
    df.to_csv(target, index=False)
    truth = int(pd.read_csv(target)["v"].sum())
    out = _session(nb_runner)
    assert f"S030 total: {truth}" in out, f"sales kept the old S030 file (truth {truth}): {out[-200:]}"

"""The output paths notebooks actually write to must resolve (round 21, R5).

``fig.savefig(OUT / 'chart.png')`` was "unresolvable" -- only a literal or a
bare name was -- so the reconstruction scope gate never suppressed it and a
chart nothing reads was re-drawn for every downstream cell.
"""
import os
from pathlib import Path

from cash.notebook.cacheability import statement_read_paths, statement_written_paths

NS = {"OUT": Path("out"), "DATA": Path("data"), "name": "x.csv", "k": 3, "REGION": None}


def _one(code):
    paths = statement_written_paths(code, namespace=NS)
    assert paths is not None, code
    (path,) = paths
    return os.path.normpath(path)


def test_the_common_spellings_resolve():
    assert _one("fig.savefig(OUT / 'overview.png', dpi=40)") == os.path.normpath("out/overview.png")
    assert _one("df.to_csv(os.path.join(OUT, name))") == os.path.normpath("out/x.csv")
    assert _one("df.to_csv(Path(OUT, 'a', name))") == os.path.normpath("out/a/x.csv")
    assert _one("df.to_csv(f'{OUT}/t_{k}.csv')") == os.path.normpath("out/t_3.csv")
    assert statement_read_paths("pd.read_csv(DATA / 'products.csv')", namespace=NS) == {
        os.path.join("data", "products.csv")}


def test_computed_paths_still_do_not_resolve():
    """Control: anything genuinely computed stays unknown, so the gate stays off."""
    assert statement_written_paths("df.to_csv(OUT / compute())", namespace=NS) is None
    assert statement_written_paths("df.to_csv(f'{OUT}/t_{k:03d}.csv')", namespace=NS) is None
    assert statement_written_paths("df.to_csv(f'{OUT}/{unknown}.csv')", namespace=NS) is None
    assert statement_written_paths("df.to_csv(OUT / name.upper())", namespace=NS) is None

"""The output paths notebooks actually write to must resolve.

``fig.savefig(OUT / 'chart.png')`` was "unresolvable" -- only a literal or a
bare name was -- so the reconstruction scope gate never suppressed it and a
chart nothing reads was re-drawn for every downstream cell.
"""

import os
from pathlib import Path

from cash.analysis.namespace_effects import statement_read_paths, statement_written_paths

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
        os.path.join("data", "products.csv")
    }


def test_computed_paths_still_do_not_resolve():
    """Control: anything genuinely computed stays unknown, so the gate stays off."""
    assert statement_written_paths("df.to_csv(OUT / compute())", namespace=NS) is None
    assert statement_written_paths("df.to_csv(f'{OUT}/t_{k:03d}.csv')", namespace=NS) is None
    assert statement_written_paths("df.to_csv(f'{OUT}/{unknown}.csv')", namespace=NS) is None
    assert statement_written_paths("df.to_csv(OUT / name.upper())", namespace=NS) is None


def test_a_read_over_a_list_of_paths_resolves_to_its_elements():
    """``pd.concat([pd.read_csv(f) for f in TF])`` read as
    "unknown", so a cell doing it re-drew an unrelated stale chart above."""
    ns = dict(NS, TF=[Path("a.csv"), "b.csv"])
    want = {"a.csv", "b.csv"}
    assert statement_read_paths("t = pd.concat([pd.read_csv(f) for f in TF])", namespace=ns) == want
    assert statement_read_paths("t = [pd.read_csv(f) for f in sorted(TF)]", namespace=ns) == want
    assert statement_read_paths("for f in TF:\n    parts.append(pd.read_csv(f))", namespace=ns) == want
    assert statement_read_paths("t = [pd.read_csv(f) for f in [DATA / 'a.csv', 'b.csv']]", namespace=NS) == {
        os.path.join("data", "a.csv"),
        "b.csv",
    }


def test_a_read_over_an_unknown_list_stays_unknown():
    """Control: a list whose elements are not all paths, or a computed one."""
    ns = dict(NS, TF=[Path("a.csv"), 3], G=(p for p in ()))
    assert statement_read_paths("t = [pd.read_csv(f) for f in TF]", namespace=ns) is None
    assert statement_read_paths("t = [pd.read_csv(f) for f in G]", namespace=ns) is None
    assert statement_read_paths("t = [pd.read_csv(f) for f in glob.glob('*.csv')]", namespace=ns) is None
    assert statement_read_paths("t = [pd.read_csv(f.name) for f in TF]", namespace=dict(NS, TF=[Path("a.csv")])) is None


def test_a_loop_variable_also_bound_elsewhere_stays_unknown():
    ns = dict(NS, TF=[Path("a.csv")])
    assert (
        statement_read_paths(
            "for f in TF:\n    x = pd.read_csv(f)\nfor f in glob.glob('*.csv'):\n    y = pd.read_csv(f)", namespace=ns
        )
        is None
    )
    assert (
        statement_read_paths("for f in TF:\n    f = f.with_suffix('.tsv')\n    x = pd.read_csv(f)", namespace=ns)
        is None
    )


def test_deleting_the_entries_of_a_listed_folder_writes_that_folder():
    """``for old in CHARTS.glob('*.png'): old.unlink()`` clears a chart
    folder before drawing. Unresolved, the scope gate could never rule it out,
    and every unrelated cell re-ran the chart cell first."""
    out = {os.path.normpath("out")}
    for code in (
        "for old in OUT.glob('*.png'):\n    old.unlink()",
        "for old in sorted(OUT.rglob('*.png')):\n    old.unlink(missing_ok=True)",
        "for old in OUT.iterdir():\n    os.remove(old)",
        "[p.unlink() for p in OUT.glob('*.png')]",
        "for f in glob.glob('out/*.png'):\n    os.unlink(f)",
        "for f in glob.glob(f'{OUT}/*.png'):\n    os.remove(f)",
    ):
        paths = statement_written_paths(code, namespace=NS)
        assert paths is not None and {os.path.normpath(p) for p in paths} == out, code
    assert statement_written_paths("Path('out/a.png').unlink()", namespace=NS) == {"out/a.png"}


def test_a_deletion_the_listing_does_not_pin_down_stays_unknown():
    """Control: a deleted path that is not simply an entry of a resolved listing."""
    for code in (
        "for old in FILES:\n    old.unlink()",
        "for old in OUT.glob('*.png'):\n    old.with_suffix('.svg').unlink()",
        "for old in OUT.glob('*.png'):\n    old = Path('/elsewhere')\n    old.unlink()",
        "for old in OUT.glob('*.png'):\n    old.unlink()\nfor old in SRC.glob('*.png'):\n    old.unlink()",
        "for old in unknown.glob('*.png'):\n    old.unlink()",
        "for f in os.listdir(OUT):\n    os.remove(os.path.join(OUT, f))",
        "for f in glob.glob('*.png', root_dir=OUT):\n    os.remove(f)",
    ):
        assert statement_written_paths(code, namespace=NS) is None, code


def test_a_folder_listing_reads_the_folder():
    """``sorted(os.listdir(OUT))`` read nothing, as the planner saw it, so a
    writer into OUT whose input was edited was left alone for the cell
    listing it."""
    out = {os.path.normpath("out")}
    for code in (
        "names = sorted(os.listdir(OUT))",
        "names = [e.name for e in os.scandir(OUT)]",
        "names = sorted(p.name for p in OUT.glob('*.txt'))",
        "names = list(OUT.rglob('*.txt'))",
        "names = list(OUT.iterdir())",
        "names = glob.glob('out/*.txt')",
    ):
        paths = statement_read_paths(code, namespace=NS)
        assert paths is not None and {os.path.normpath(p) for p in paths} == out, code
    assert statement_read_paths("names = os.listdir()", namespace=NS) == {os.curdir}


def test_a_listing_of_an_unknown_folder_is_an_unknown_read():
    """Control: the folder must resolve, or the read set is unknown and no
    writer is ruled out."""
    assert statement_read_paths("names = os.listdir(somewhere())", namespace=NS) is None
    assert statement_read_paths("names = list(unknown.glob('*'))", namespace=NS) is None

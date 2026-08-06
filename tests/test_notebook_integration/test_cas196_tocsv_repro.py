"""CAS-196: df.to_csv() must not bump df's lineage (no re-fired audit append).

``df.to_csv(path)`` READS the frame and writes a file; it does not mutate ``df``.
The receiver-mutation classifier used to assume-mutate a DataFrame receiver
(``compute_hash`` samples large frames, so it can't prove purity), making the
write a spurious *producer* of ``df``. Upstream reconstruction then re-scheduled
the write to rebuild ``df`` and re-fired a non-idempotent append, corrupting the
audit log. The fix classifies ``to_csv`` (and siblings) as receiver-read-only.
"""
import pytest

pytest.importorskip("pandas")

pytestmark = pytest.mark.files


def _rows(path):
    try:
        with open(path) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def _restart(nb_runner):
    import asyncio
    nb_runner.restart()
    nb_runner._inject_notebook_path()


@pytest.mark.timeout(150)
def test_tocsv_append_not_refired_by_downstream_reader(nb_runner, tmp_path):
    audit = (tmp_path / "audit.csv").as_posix()
    nb_runner.create_notebook([
        "import pandas as pd\nimport cash\n%cash_on\n%cash_badge print",
        "df = pd.DataFrame({'x': [1, 2, 3]})",
        # Non-idempotent append-mode ETL audit write.
        f"df.to_csv('{audit}', mode='a', header=False, index=False)\n"
        "print('audited')",
        # A downstream reader that depends on df.
        "s = int(df['x'].sum())\nprint('SUM:', s)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _rows(audit) == 3, "one run-all should append exactly the 3 df rows"

    # Restart, replay setup + the df producer, then run ONLY the downstream
    # reader. Reconstructing df must NOT re-fire the to_csv append.
    _restart(nb_runner)
    nb_runner.run_cells([1, 2])
    assert _rows(audit) == 3, "setup must not touch the audit file"

    nb_runner.run_cells([4])
    out = nb_runner.get_output(4)
    assert _rows(audit) == 3, (
        f"downstream reader re-fired the df.to_csv append during reconstruction: "
        f"audit grew to {_rows(audit)} rows (expected 3). df.to_csv bumped df's "
        f"lineage and became a spurious producer of df (CAS-196). Badge:\n{out}"
    )
    assert "SUM: 6" in out, f"reader served wrong data after reconstruction: {out}"

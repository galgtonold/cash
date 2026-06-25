"""Regression: a skipped/restored import must still rebind its name after a
kernel restart.

A cache hit for an `import X` statement takes the restore path, but restoring
cannot rebind a *module* object. On a fresh kernel (after a restart) the bound
name is therefore missing, so a later statement that misses (e.g. its file dep
changed) and uses the name raised ``NameError: name 'X' is not defined``. The
processor now forces re-execution of a pure-import whose names are absent from
the namespace, rebinding them cheaply.
"""
import time

import pytest

pytestmark = [pytest.mark.files, pytest.mark.restore]


def test_import_rebinds_when_downstream_recomputes_after_restart(nb_runner, tmp_path):
    """Import and the file read live in separate cells. After a restart the
    import is restored (not executed) and the file-read cell recomputes because
    the file changed - it must still see the rebound module."""
    csv = tmp_path / "data.csv"
    csv_str = str(csv).replace("\\", "/")
    csv.write_text("v\n1\n2\n3\n")

    nb_runner.create_notebook([
        "import pandas as pd",
        f"df = pd.read_csv('{csv_str}')",
        "s = int(df['v'].sum())\nprint(f's = {s}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "s = 6" in nb_runner.get_output(3)

    # Restart, change the file so the read cell must recompute.
    nb_runner.shutdown()
    csv.write_text("v\n10\n20\n30\n")
    time.sleep(0.1)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "s = 60" in nb_runner.get_output(3), (
        "downstream recompute after restart hit NameError (import not rebound)"
    )


def test_import_only_cell_unchanged_file_still_works_after_restart(nb_runner, tmp_path):
    """Sanity: when nothing changed, a restart + full re-run still produces the
    right answer (the import is rebound on re-execution, downstream restores)."""
    csv = tmp_path / "data.csv"
    csv_str = str(csv).replace("\\", "/")
    csv.write_text("v\n4\n5\n6\n")

    nb_runner.create_notebook([
        "import pandas as pd",
        f"df = pd.read_csv('{csv_str}')",
        "s = int(df['v'].sum())\nprint(f's = {s}')",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "s = 15" in nb_runner.get_output(3)

    nb_runner.shutdown()
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "s = 15" in nb_runner.get_output(3)

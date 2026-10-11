"""``pd.read_parquet(folder)`` in a cell sees an edit under a linked sub-folder.

pandas reads the files under ``dd/sub -> ../other/sub``; cash expanded the
folder without following the link, so the next Run All after an edit there
showed the old total.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(120)]


def test_an_edit_under_a_linked_sub_folder_is_seen_on_run_all(nb_runner, tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    dd = tmp_path / "dd"
    dd.mkdir()
    pq.write_table(pa.table({"x": [1]}), dd / "p1.parquet")
    other = tmp_path / "other" / "sub"
    other.mkdir(parents=True)
    pq.write_table(pa.table({"x": [10]}), other / "p2.parquet")
    try:
        os.symlink(other, dd / "sub", target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"cannot make a directory symlink here: {exc}")
    path = str(dd).replace("\\", "/")
    nb_runner.create_notebook(
        [
            "import pandas as pd, time",
            f"df = pd.read_parquet('{path}')\ntime.sleep(0.2)",
            "print(int(df.x.sum()))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "11"

    pq.write_table(pa.table({"x": [20]}), other / "p2.parquet")
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "21"

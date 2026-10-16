"""Three cells that each read ``data.csv``, change it and write it back are
named on the first Run All (``NOTEBOOK-FILE-REWRITE-CHAIN``), once.

Each re-runs the ones above it, so a Run All runs them six times.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.files, pytest.mark.timeout(180)]

CODE = "[NOTEBOOK-FILE-REWRITE-CHAIN]"


def _cells(path: str) -> list[str]:
    def step(n: int) -> str:
        return f"df = pd.read_csv(P)\ndf['x'] = df['x'] + {n}\ndf.to_csv(P, index=False)"

    return [
        f"import pandas as pd\nP = {path!r}\npd.DataFrame({{'x': [0, 1]}}).to_csv(P, index=False)",
        step(1),
        step(10),
        step(100),
        "print(pd.read_csv(P)['x'].tolist())",
    ]


def test_three_rewriting_cells_are_named_once(nb_runner, tmp_path):
    path = str(tmp_path / "data.csv").replace("\\", "/")
    nb_runner.create_notebook(_cells(path))
    nb_runner.start_kernel()
    nb_runner.run_all()

    outputs = [nb_runner.get_raw_output(n) for n in range(1, 6)]
    assert sum(CODE in out for out in outputs) == 1, outputs
    third = outputs[2]
    assert CODE in third, third
    assert "cells 2 and 3" in third and "data.csv" in third, third
    assert CODE not in outputs[1], "a single rewriting cell was named"

    nb_runner.run_all()
    assert not any(CODE in nb_runner.get_raw_output(n) for n in range(1, 6)), "named again"

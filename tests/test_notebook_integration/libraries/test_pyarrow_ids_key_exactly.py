"""Two pyarrow-backed frames whose large ids differ get their own results in a
loop, on the first run: their ids are keyed exactly, nulls included."""

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.timeout(120)]


def test_a_loop_over_frames_with_large_ids_and_a_null(nb_runner):
    pytest.importorskip("pyarrow")
    nb_runner.create_notebook(
        [
            "import time\nimport pandas as pd, pyarrow as pa\n"
            f"def slow_max(df):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return int(df['id'].max())",
            "frames = [pd.DataFrame({'id': pd.array([1234567890123456789, None], dtype='int64[pyarrow]')}),\n"
            "          pd.DataFrame({'id': pd.array([1234567890123456700, None], dtype='int64[pyarrow]')})]",
            "out = []\nfor f in frames:\n    out.append(slow_max(f))\nprint(out)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == "[1234567890123456789, 1234567890123456700]"

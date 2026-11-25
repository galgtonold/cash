"""A script whose cached function returned a pandas table ends as quietly
as the plain script.

While a result is looked through, cash remembers what it read in the Arrow
text columns and drops the note when the column is freed. A column still
alive when Python exits is freed after the modules' globals were cleared;
the clean-up must not fail there, or Python prints "Exception ignored"
with a traceback at the end of a run that went well.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

TEXT_COLUMN = """
    import sys, warnings
    import numpy as np, pandas as pd
    import cash
    warnings.simplefilter("ignore")

    @cash.cache
    def load(n):
        return pd.DataFrame({"a": np.arange(n), "s": np.array(["x", "yy", "zzz", "w"])[np.arange(n) % 4]})

    df = load(int(sys.argv[1]))
    print("rows", len(df))
"""

ARROW_TEXT_COLUMN = """
    import sys, warnings
    import numpy as np, pandas as pd
    import cash
    warnings.simplefilter("ignore")

    @cash.cache
    def load(n):
        text = pd.array(np.array(["id 1234567", "yy", "0xabcdef", "w"])[np.arange(n) % 4], dtype="string[pyarrow]")
        return pd.DataFrame({"a": np.arange(n), "s": text})

    df = load(int(sys.argv[1]))
    print("rows", len(df))
"""

FLOAT_COLUMNS = """
    import sys, warnings
    import numpy as np, pandas as pd
    import cash
    warnings.simplefilter("ignore")

    @cash.cache
    def make(n):
        return pd.DataFrame(np.random.default_rng(0).random((n, 10)), columns=[f"c{i}" for i in range(10)])

    df = make(int(sys.argv[1]))
    print("rows", len(df))
"""


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    ("source", "rows"),
    [
        pytest.param(TEXT_COLUMN, 300_000, id="text-column"),
        pytest.param(ARROW_TEXT_COLUMN, 300_000, id="arrow-text-column"),
        pytest.param(FLOAT_COLUMNS, 400_000, id="float-columns"),
    ],
)
def test_the_script_prints_nothing_on_stderr_when_it_exits(tmp_path, source, rows):
    (tmp_path / "main.py").write_text(textwrap.dedent(source), encoding="utf-8")
    for run in ("computed", "served"):
        done = run_python("main.py", str(rows), cwd=tmp_path)
        assert done.stdout.strip() == f"rows {rows}", run
        assert done.stderr == "", f"{run}: {done.stderr}"


def test_a_failing_callback_is_still_reported_while_the_process_runs():
    """Positive control for the guard: outside shutdown a callback's
    error is not swallowed."""
    from cash import _weak

    class Thing:
        pass

    def boom(_ref):
        raise RuntimeError("callback failed")

    thing = Thing()
    reference = _weak.ref(thing, boom)
    with pytest.raises(RuntimeError, match="callback failed"):
        _weak.callback_of(reference)(reference)

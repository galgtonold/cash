"""A library function that never writes into its arguments has them hashed
by its key alone, not again before and after it runs.

``df = pd.concat(df_array)`` over 87,000 small frames hashed every frame
before and after the call, to see whether ``concat`` changed one: 100 s on
top of a loop plain Python ran in 87 s. ``concat``, ``merge`` and numpy's
reductions only read what they are given (`reads_its_arguments_only`).
"""

from __future__ import annotations

import pytest

from cash.notebook import call_effects
from tests._cell_driver import run_cash_cell

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


@pytest.fixture
def hashed_frames(monkeypatch):
    """The frames each call's arguments held when they were hashed."""
    import cash.notebook.call_unit as call_unit

    frames = []
    real = call_unit.hash_args

    def counting(args, kwargs, *rest, **options):
        for value in (*args, *kwargs.values()):
            frames.extend(v for v in (value if isinstance(value, list) else [value]) if isinstance(v, pd.DataFrame))
        return real(args, kwargs, *rest, **options)

    monkeypatch.setattr(call_unit, "hash_args", counting)
    return frames


def test_the_read_only_functions_are_known_by_identity():
    assert call_effects.reads_its_arguments_only(pd.concat)
    assert call_effects.reads_its_arguments_only(np.quantile)

    def concat(frames):
        frames.append(None)

    assert not call_effects.reads_its_arguments_only(concat)
    assert not call_effects.reads_its_arguments_only(pd.DataFrame.sort_values)
    assert not call_effects.reads_its_arguments_only(len)


@pytest.mark.parametrize("read_only", [True, False])
def test_concat_does_not_hash_the_frames_it_joins(cash_magics, hashed_frames, monkeypatch, read_only):
    if not read_only:
        import cash.notebook.call_unit as call_unit

        monkeypatch.setattr(call_unit, "reads_its_arguments_only", lambda fn: False)
    run_cash_cell(cash_magics, "import pandas as pd\nframes = [pd.DataFrame({'a': [i, i]}) for i in range(30)]")
    hashed_frames.clear()
    run_cash_cell(cash_magics, "df = pd.concat(frames, ignore_index=True)")
    assert cash_magics.shell.user_ns["df"]["a"].tolist() == [i for i in range(30) for _ in range(2)]
    if read_only:
        assert not hashed_frames
    else:
        assert len(hashed_frames) >= 60  # the control: before and after, each frame

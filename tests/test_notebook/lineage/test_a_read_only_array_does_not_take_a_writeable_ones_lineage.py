"""An array rebuilt with the same values but read-only is not the value an
earlier writeable one was, so it does not take that one's lineage.

``arr = np.array(df["a"])`` edited to ``arr = df["a"].to_numpy()`` (read-only
in pandas 3, same values) took the old lineage, and the next cell's
``arr[arr < 2] = 0`` was served from the cache instead of raising
"assignment destination is read-only". The kernel-level case is
``tests/test_notebook_integration/reruns/test_a_read_only_array_rebuilt_equal_still_refuses_writes.py``.
"""

import pytest

from cash.notebook.value_cutoff import ValueCutoff

np = pytest.importorskip("numpy")


def _content(value):
    return value.tobytes().hex()


def test_only_an_equal_array_that_is_as_writeable_takes_the_lineage():
    cutoff = ValueCutoff()
    writeable = np.array([1.0, 2.0])
    read_only = np.array([1.0, 2.0])
    read_only.setflags(write=False)
    assert cutoff.lineage_for(writeable, "L1", _content) == "L1"
    assert cutoff.lineage_for(read_only, "L2", _content) == "L2"
    # Positive control: an equal writeable array still takes the first one's.
    assert cutoff.lineage_for(np.array([1.0, 2.0]), "L3", _content) == "L1"
    # And an equal read-only one the read-only one's.
    again = np.frombuffer(np.array([1.0, 2.0]).tobytes())
    assert cutoff.lineage_for(again, "L4", _content) == "L2"


def test_a_frame_over_a_read_only_block_is_not_one_over_a_writeable_block():
    pd = pytest.importorskip("pandas")
    cutoff = ValueCutoff()
    frame = pd.DataFrame({"a": [1.0, 2.0]})

    def content(value):
        return str(value.to_dict())

    assert cutoff.lineage_for(frame, "F1", content) == "F1"
    blocks = frame._mgr.arrays
    if not blocks or not isinstance(blocks[0], np.ndarray):
        pytest.skip("this pandas keeps the column in no numpy block")
    held = frame.copy()
    held._mgr.arrays[0].setflags(write=False)
    assert cutoff.lineage_for(held, "F2", content) == "F2"
    assert cutoff.lineage_for(frame.copy(), "F3", content) == "F1"

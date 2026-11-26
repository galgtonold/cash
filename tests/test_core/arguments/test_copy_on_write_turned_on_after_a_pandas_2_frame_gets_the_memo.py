"""Under pandas 2, a frame met before copy-on-write is on does not keep the
frame memo off once it is.

Copy-on-write, which the frame memo needs, is opt-in before pandas 3. The
verdict on whether this pandas may use the memo (its version, the start-up
check) is kept for the process, but whether copy-on-write is on is read
again whenever the cached answer of ``arg_hashing.is_cow_pandas`` is
forgotten: a frame met with it off must not decide for a later frame met
with it on.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from cash import _pandas_compat
from cash.decorator import arg_hashing

pytestmark = pytest.mark.skipif(
    int(pd.__version__.split(".")[0]) >= 3, reason="copy-on-write is the only mode from pandas 3 on"
)


def _forget_verdict() -> None:
    _pandas_compat.reset_frame_memo_state()
    arg_hashing._COW_PANDAS = None
    arg_hashing.FRAME_DIGESTS.clear()


@pytest.fixture
def undecided():
    """The process as before its first frame, with copy-on-write off; put
    back so afterwards."""
    before = pd.get_option("mode.copy_on_write")
    pd.set_option("mode.copy_on_write", False)
    _forget_verdict()
    yield
    pd.set_option("mode.copy_on_write", before)
    _forget_verdict()


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})


def test_a_frame_met_without_copy_on_write_gets_no_memo(undecided):
    assert not arg_hashing.is_cow_pandas(_frame())


def test_copy_on_write_turned_on_later_gets_the_memo(undecided):
    assert not arg_hashing.is_cow_pandas(_frame())
    pd.set_option("mode.copy_on_write", True)
    arg_hashing._COW_PANDAS = None
    assert arg_hashing.is_cow_pandas(_frame())
    assert _pandas_compat.frame_memo_state() == (True, "")

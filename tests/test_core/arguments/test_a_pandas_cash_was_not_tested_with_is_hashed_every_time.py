"""A pandas cash cannot vouch for has its frames hashed on every call.

The frame memo does not read a frame again while it can prove that nothing
wrote to it, and the proof is a list of the routes by which pandas lets a
frame's memory be written. pandas is not a dependency of cash, so the
user's ``pip install -U pandas`` decides which pandas that list meets, and
a route that is not on it is a stale result.

So the first frame that reaches cash decides for the process
(``cash._pandas_compat.frame_memo_usable``): a pandas newer than the
newest one tested, or one on which the memo misses a write in the
start-up check, gets no memo. Every frame is then read in full each time,
which is always right, and ``CACHE-PANDAS-UNTESTED`` says so once. The key
of a frame is the same either way, so an entry stored with the memo on is
found with it off.
"""

from __future__ import annotations

import sys
import warnings

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from cash import _pandas_compat
from cash.content_hashers import builtin_hash
from cash.decorator import arg_hashing
from tests._scripts import run_python

CODE = "CACHE-PANDAS-UNTESTED"

pytestmark = pytest.mark.skipif(
    int(pd.__version__.split(".")[0]) < 3, reason="copy-on-write, which the frame memo needs, is pandas 3's"
)


def _forget_verdict() -> None:
    _pandas_compat.reset_frame_memo_state()
    arg_hashing._COW_PANDAS = None
    arg_hashing.FRAME_DIGESTS.clear()


@pytest.fixture
def undecided():
    """The process as before its first frame; put back so afterwards."""
    _forget_verdict()
    yield
    _forget_verdict()


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})


def _one_minor_older() -> tuple[int, int]:
    major, minor = _pandas_compat.pandas_version()
    return (major, minor - 1) if minor else (major - 1, 99)


def _told(caught) -> list:
    return [w for w in caught if getattr(w.message, "code", None) == CODE]


def test_the_installed_pandas_is_one_the_memo_was_tested_with():
    """Fails the day a newer pandas is installed. Run
    tests/test_core/arguments/test_a_frame_written_by_any_route_is_hashed_again.py
    against it (the nightly workflow does, with pre-releases), then raise
    ``cash._pandas_compat.NEWEST_TESTED``."""
    assert _pandas_compat.pandas_version() <= _pandas_compat.NEWEST_TESTED


def test_the_memo_is_used_with_a_tested_pandas(undecided):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert arg_hashing.is_cow_pandas(_frame())
    assert _pandas_compat.frame_memo_state() == (True, "")
    assert not _told(caught)


def test_the_check_leaves_nothing_in_the_memo_and_nothing_on_screen(undecided, capsys):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        arg_hashing.is_cow_pandas(_frame())
    assert len(arg_hashing.FRAME_DIGESTS) == 0
    assert not arg_hashing._PROBE_MEMO
    assert not caught, [str(w.message) for w in caught]
    assert capsys.readouterr() == ("", "")


class TestANewerPandas:
    @pytest.fixture(autouse=True)
    def _newer(self, undecided, monkeypatch):
        monkeypatch.setattr(_pandas_compat, "NEWEST_TESTED", _one_minor_older())

    def test_gets_no_memo_and_says_so_once(self, disk_cash):
        @disk_cash.cache
        def total(df):
            return float(df["a"].sum())

        frame = _frame()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert total(frame) == 6.0
            assert total(frame) == 6.0
            assert total(_frame()) == 6.0
        told = _told(caught)
        assert len(told) == 1
        text = str(told[0].message)
        assert "newer than the newest pandas this cash was tested with" in text
        assert pd.__version__.rsplit(".", 1)[0] in text
        assert not arg_hashing.is_cow_pandas(frame)
        assert len(arg_hashing.FRAME_DIGESTS) == 0

    def test_reads_the_frame_on_every_call(self, disk_cash, monkeypatch):
        reads = []
        real = arg_hashing.builtin_hash

        def counting(value, *args, **kwargs):
            if isinstance(value, pd.DataFrame):
                reads.append(1)
            return real(value, *args, **kwargs)

        monkeypatch.setattr(arg_hashing, "builtin_hash", counting)

        @disk_cash.cache
        def total(df):
            return float(df["a"].sum())

        frame = _frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(3):
                total(frame)
        assert len(reads) >= 3

    def test_still_serves_what_the_frame_holds_now(self, disk_cash):
        @disk_cash.cache
        def total(df):
            return float(df["a"].sum())

        frame = _frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert total(frame) == 6.0
            frame.iloc[0, 0] = 10.0
            assert total(frame) == 15.0

    def test_a_filter_that_raises_does_not_fail_the_call(self, disk_cash):
        @disk_cash.cache
        def total(df):
            return float(df["a"].sum())

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert total(_frame()) == 6.0
        assert _pandas_compat.frame_memo_state()[0] is False


def test_a_frame_keys_the_same_with_the_memo_and_without(undecided, disk_cash, monkeypatch):
    runs = []

    @disk_cash.cache
    def total(df):
        runs.append(1)
        return float(df["a"].sum())

    frame = _frame()
    assert total(frame) == 6.0
    with_memo = arg_hashing.cow_frame_digest(frame)
    assert with_memo == builtin_hash(frame)
    assert _pandas_compat.frame_memo_state() == (True, "")

    _forget_verdict()
    monkeypatch.setattr(_pandas_compat, "NEWEST_TESTED", _one_minor_older())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert total(frame) == 6.0
        assert total(_frame()) == 6.0
    assert _pandas_compat.frame_memo_state()[0] is False
    assert len(runs) == 1, "the entry stored with the memo on was not found with it off"


class TestTheStartUpCheck:
    def test_finds_a_route_the_memo_does_not_know(self, undecided, monkeypatch):
        """A write past every accessor, straight into the block's array: the
        memo cannot see it, and the check must say so."""
        unknown = ("the block's own array", "num", "df._mgr.arrays[0][0] = 9.0")
        monkeypatch.setattr(_pandas_compat, "WRITE_ROUTES", (*_pandas_compat.WRITE_ROUTES, unknown))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert not arg_hashing.is_cow_pandas(_frame())
        usable, reason = _pandas_compat.frame_memo_state()
        assert not usable
        assert "a write through the block's own array goes unnoticed" in reason
        told = _told(caught)
        assert len(told) == 1 and "start-up check" in str(told[0].message)

    def test_passes_a_route_pandas_refuses(self, undecided, monkeypatch):
        refused = ("a refused write", "num", "raise ValueError('assignment destination is read-only')")
        monkeypatch.setattr(_pandas_compat, "WRITE_ROUTES", (*_pandas_compat.WRITE_ROUTES, refused))
        assert arg_hashing.is_cow_pandas(_frame())

    def test_fails_when_pandas_keeps_its_data_elsewhere(self, undecided, monkeypatch):
        def gone(obj):
            raise AttributeError("'DataFrame' object has no attribute '_mgr'")

        monkeypatch.setattr(_pandas_compat, "blocks", gone)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert not arg_hashing.is_cow_pandas(_frame())
        assert "pandas does not keep a frame's blocks" in _pandas_compat.frame_memo_state()[1]
        assert len(_told(caught)) == 1

    def test_fails_when_the_memo_never_engages(self, undecided):
        problems = _pandas_compat.self_test(pd, np, lambda frame: False, lambda frame: False)
        assert problems == ["the memo does not take an untouched frame"]

    def test_every_route_really_writes_or_is_refused(self):
        """Each route changes the frame under this pandas, or raises: none
        passes the check because it quietly does nothing."""
        frames = _pandas_compat.probe_frames(pd, np)
        idle = []
        for (name, kind, _statements), route in zip(
            _pandas_compat.WRITE_ROUTES, _pandas_compat.compile_routes().values()
        ):
            pristine = _pandas_compat.probe_frames(pd, np)[kind]
            frame = frames[kind].copy()
            frame.index = frame.index.copy()  # its own labels: a copy shares its index
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    route(pd, np, frame)
                except Exception:
                    continue
            same = frame.equals(pristine) and frame.index.name == pristine.index.name
            same = same and frame.attrs == pristine.attrs
            same = same and frame.flags.allows_duplicate_labels == pristine.flags.allows_duplicate_labels
            if same:
                idle.append(name)
        # Copies under pandas 3.1; kept for a pandas that hands out the memory.
        assert set(idle) <= {"Index.values", "asi8"}, idle

    def test_runs_once_per_process(self, undecided, monkeypatch):
        runs = []
        real = _pandas_compat.self_test

        def counting(*args):
            runs.append(1)
            return real(*args)

        monkeypatch.setattr(_pandas_compat, "self_test", counting)
        for _ in range(5):
            arg_hashing.is_cow_pandas(_frame())
        arg_hashing.is_cow_pandas(pd.Series([1.0]))
        assert len(runs) == 1


def test_nothing_is_checked_before_a_frame_arrives(undecided, disk_cash):
    @disk_cash.cache
    def double(x):
        return 2 * x

    assert double(2) == 4
    assert not arg_hashing.is_cow_pandas([1, 2, 3])
    assert _pandas_compat.frame_memo_state() is None


def test_a_program_without_pandas_never_loads_it(tmp_path):
    script = (
        "import sys\n"
        "import cash\n"
        "from cash import _pandas_compat\n"
        "@cash.cache\n"
        "def double(x):\n"
        "    return 2 * x\n"
        "assert double(2) == 4 and double(2) == 4\n"
        "assert _pandas_compat.pandas_version() is None\n"
        "assert _pandas_compat.frame_memo_state() is None\n"
        "assert 'pandas' not in sys.modules\n"
        "print('ok')\n"
    )
    assert run_python("-c", script, cwd=tmp_path).stdout.strip() == "ok"


@pytest.mark.parametrize(
    "text, expected",
    [("3.1.1", (3, 1)), ("3.2.0rc1", (3, 2)), ("4.0.0.dev0+123.gabc", (4, 0)), ("3.10", (3, 10)), ("2.3.3", (2, 3))],
)
def test_the_version_is_read_as_major_and_minor(monkeypatch, text, expected):
    monkeypatch.setattr(pd, "__version__", text)
    assert _pandas_compat.pandas_version() == expected


def test_an_unreadable_version_is_an_untested_pandas(undecided, monkeypatch):
    monkeypatch.setattr(pd, "__version__", "unknown")
    assert _pandas_compat.pandas_version() is None
    assert not arg_hashing.is_cow_pandas(_frame())


def test_the_module_is_imported_without_pandas():
    assert "pandas" not in _pandas_compat.__dict__
    assert sys.modules["cash._pandas_compat"] is _pandas_compat

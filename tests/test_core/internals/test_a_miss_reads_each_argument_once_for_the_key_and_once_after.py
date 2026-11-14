"""What a missed call pays for its arguments and its state segment.

A miss digests each argument's content twice at most: once for the key and
once after the body, to see whether the body changed it in place. Storing the
result, checking what it shares with the arguments and copying it into the
memory tier read no argument again, and never digest the result itself.

A pandas body that clips a series runs ``warnings.catch_warnings(record=True)``
(pandas' ``rewrite_warning`` around ``align``), which swaps the warnings
module's display hooks for a recorder and back. That moved the extension's
world epoch on every call, so a pandas-heavy function rebuilt its whole state
segment on every miss. The hooks are quiet now unless scanned code names them,
and a key set to the object it already holds is no change at all.
"""

from __future__ import annotations

import collections
import warnings

import pytest

from cash import Cash, content_hashers
from cash.decorator import arg_hashing, state_memo

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

pytestmark = [
    pytest.mark.core,
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

TIMES = pd.date_range("2020-01-01", periods=5000, freq="1h", tz="Etc/GMT+7")


def clear_sky(times, weather, scale, offset, albedo=None):
    out = pd.DataFrame(
        {
            "ghi": weather["ghi"].clip(lower=0) * scale + offset,
            "dni": np.sqrt(weather["dni"].clip(lower=0)) - scale,
        },
        index=times,
    )
    out["hour"] = times.hour
    if albedo is not None:
        out["reflected"] = out["ghi"].to_numpy() * albedo
    return out


def doubles_in_place(times, weather, scale, offset):
    weather.iloc[0, 0] = weather.iloc[0, 0] * 2 + 1
    return weather["ghi"] * scale + offset


def _weather():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"ghi": rng.random(len(TIMES)) * 900, "dni": rng.random(len(TIMES)) * 800}, index=TIMES)


@pytest.fixture
def digests(monkeypatch):
    """``id(value) -> times its content was read into a digest``: through
    `builtin_hash` (frames, arrays) or `hash_pandas_index` (an index, which
    the payload walk keys by its labels)."""
    counts: collections.Counter = collections.Counter()
    real_builtin, real_index = arg_hashing.builtin_hash, content_hashers.hash_pandas_index

    def builtin(value):
        digest = real_builtin(value)
        if digest is not None:
            counts[id(value)] += 1
        return digest

    def index(value):
        counts[id(value)] += 1
        return real_index(value)

    monkeypatch.setattr(arg_hashing, "builtin_hash", builtin)
    monkeypatch.setattr(content_hashers, "hash_pandas_index", index)
    return counts


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path), register_magic=False)


def test_a_miss_digests_each_argument_for_the_key_and_after_the_body_only(c, digests):
    f = c.cache(clear_sky)
    albedo = np.full(len(TIMES), 0.2)
    f(TIMES, _weather(), 0.5, 0.0, albedo)  # analysis, first sights: not what is counted
    for offset in (1.0, 2.0, 3.0):
        weather = _weather()  # a frame the memo has never seen: read in full
        digests.clear()
        misses = f.cache_info()["misses"]
        result = f(TIMES, weather, 0.7, offset, albedo)
        assert f.cache_info()["misses"] == misses + 1
        # The key reads each once; the check after the body may read each
        # once more. The snapshot that names a changed argument, the store,
        # the share check and the memory tier read none of them again.
        assert 1 <= digests[id(TIMES)] <= 2, "the index was read beyond the key and the mutation check"
        assert 1 <= digests[id(albedo)] <= 2, "the array was read beyond the key and the mutation check"
        # A frame is read once; the check after the body finds it unchanged
        # in the copy-on-write memo without reading it again. Before pandas
        # 3, without copy-on-write, no memo can tell the body left it alone:
        # the check reads it again, as it reads the array.
        assert digests[id(weather)] == (1 if arg_hashing.is_cow_pandas(weather) else 2)
        assert digests[id(result)] == 0, "storing the result digested it"
    assert f(TIMES, weather, 0.7, 3.0, albedo).equals(result)  # stored: served


def test_an_argument_the_body_changes_is_still_caught_and_named(c, digests):
    f = c.cache(doubles_in_place)
    weather = _weather()
    first = weather.iloc[0, 0]
    with pytest.warns(Warning, match="'weather'|weather") as caught:
        f(TIMES, weather, 0.7, 1.0)
    assert weather.iloc[0, 0] == first * 2 + 1
    assert any("weather" in str(w.message) and "times" not in str(w.message) for w in caught)
    assert f.cache_info()["misses"] == 1
    # Not served from a stored entry: the body runs and changes it again.
    f(TIMES, weather, 0.7, 1.0)
    assert weather.iloc[0, 0] == (first * 2 + 1) * 2 + 1


needs_memo = pytest.mark.skipif(not state_memo.ENABLED, reason="the state memo needs the C extension")


@needs_memo
def test_pandas_recording_its_warnings_leaves_the_state_segment_remembered(c):
    f = c.cache(clear_sky)
    weather = _weather()
    for offset in range(3):
        f(TIMES, weather, 0.5, float(offset))
    before = dict(state_memo.STATS)
    for offset in range(3, 9):
        f(TIMES, weather, 0.5, float(offset))
    assert state_memo.STATS["reused"] - before["reused"] >= 5


@needs_memo
def test_the_warning_hooks_move_the_world_only_when_swapped_for_something_new():
    from cash import _native

    cw = _native.cwatch
    Cash(backend="memory", register_magic=False).cache(clear_sky)(TIMES, _weather(), 0.5, 0.0)  # watches modules
    epoch = cw.world_epoch()
    with warnings.catch_warnings(record=True):
        warnings.warn("recorded", UserWarning, stacklevel=1)
    assert cw.world_epoch() == epoch
    hook = warnings.formatwarning
    warnings.formatwarning = hook  # the object it holds: no change
    assert cw.world_epoch() == epoch
    try:
        warnings.formatwarning = lambda *a, **k: ""  # a new callable under a name a key may call
        assert cw.world_epoch() != epoch
    finally:
        warnings.formatwarning = hook

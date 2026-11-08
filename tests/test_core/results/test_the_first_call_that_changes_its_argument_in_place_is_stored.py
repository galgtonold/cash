"""The first call seen changing an argument in place is stored, with the
state it left the argument in, when the key holds that argument by all it
holds: a NumPy array, a pandas frame, a list, dict or set of plain values,
with no ``key=``, ``__cash_key__`` or registered hasher standing in for it.

The key's hash of such an argument is its state before the call, so the
entry needs nothing more to say which argument a hit may write the stored
state into: a hit's key match says the argument handed in is in that state.
Nothing is hashed a second time for it. An argument the key holds some
other way is fingerprinted before the body from the second call on, and
the first call is not stored.
"""

from __future__ import annotations

import textwrap
import warnings

import numpy as np
import pandas as pd
import pytest

from tests._scripts import run_python


def _clip(a):
    np.clip(a, 0, 1, out=a)
    return float(a.sum())


def test_the_second_call_hits_and_leaves_the_argument_as_plain_python_does(disk_cash):
    runs = []

    @disk_cash.cache
    def clip_in_place(a):
        runs.append(1)
        return _clip(a)

    a1 = np.array([-1.0, 0.5, 2.0])
    assert clip_in_place(a1) == 1.5
    a2 = np.array([-1.0, 0.5, 2.0])
    alias = a2
    assert clip_in_place(a2) == 1.5
    assert len(runs) == 1, "the first call was not stored"
    plain = np.array([-1.0, 0.5, 2.0])
    _clip(plain)
    assert alias is a2
    assert a2.dtype == plain.dtype and a2.tolist() == plain.tolist() == [0.0, 0.5, 1.0]


def test_an_argument_holding_other_values_is_not_served(disk_cash):
    runs = []

    @disk_cash.cache
    def clip_in_place(a):
        runs.append(1)
        return _clip(a)

    clip_in_place(np.array([-1.0, 0.5, 2.0]))
    other = np.array([-1.0, 0.6, 2.0])
    assert clip_in_place(other) == 1.6
    assert other.tolist() == [0.0, 0.6, 1.0]
    assert len(runs) == 2
    again = np.array([-1.0, 0.5, 2.0])
    assert clip_in_place(again) == 1.5
    assert again.tolist() == [0.0, 0.5, 1.0]
    assert len(runs) == 2, "the first call's entry was not served to the same values"


def test_a_key_function_keeps_the_first_call_unstored(disk_cash):
    """``key=`` holds what it returns, not the argument's state: its first
    call is not stored, as before. The same body without it is."""
    keyed_runs, plain_runs = [], []

    @disk_cash.cache(key=lambda a: len(a))
    def keyed(a):
        keyed_runs.append(1)
        return _clip(a)

    @disk_cash.cache
    def plain(a):
        plain_runs.append(1)
        return _clip(a)

    for _ in range(2):
        a, b = np.array([-1.0, 2.0]), np.array([-1.0, 2.0])
        keyed(a)
        plain(b)
        assert a.tolist() == b.tolist() == [0.0, 1.0]
    assert len(keyed_runs) == 2
    assert len(plain_runs) == 1


def test_a_registered_hasher_keeps_the_first_call_unstored(tmp_path):
    """A hasher the user registered names the argument by what it returns:
    the first call is not stored, the next ones are, checked by a
    fingerprint taken before the body. A Cash with none stores the first."""
    from cash import Cash

    hashed = Cash(cache_dir=str(tmp_path / "hashed"), register_magic=False)
    hashed.register_hasher(np.ndarray, lambda a: a.tobytes().hex(), override=True)
    plain = Cash(cache_dir=str(tmp_path / "plain"), register_magic=False)
    runs = {"hashed": [], "plain": []}

    def make(cash, name):
        @cash.cache
        def work(a):
            runs[name].append(1)
            return _clip(a)

        return work

    hashed_work, plain_work = make(hashed, "hashed"), make(plain, "plain")
    for _ in range(3):
        a, b = np.array([-1.0, 2.0]), np.array([-1.0, 2.0])
        hashed_work(a)
        plain_work(b)
        assert a.tolist() == b.tolist() == [0.0, 1.0]
    assert len(runs["hashed"]) == 2
    assert len(runs["plain"]) == 1


def test_a_frame_given_a_new_column_runs_every_time(disk_cash):
    """A state of another shape cannot be written into the caller's frame:
    not stored, the first call or any other. Changing only values is."""
    grown_runs, scaled_runs = [], []

    @disk_cash.cache
    def grow(df):
        grown_runs.append(1)
        df["c"] = df["a"] * 3
        return 1

    @disk_cash.cache
    def scale(df):
        scaled_runs.append(1)
        df["a"] *= 3
        return 1

    for _ in range(3):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        grow(df)
        assert list(df.columns) == ["a", "c"] and df["c"].tolist() == [3.0, 6.0]
        df = pd.DataFrame({"a": [1.0, 2.0]})
        scale(df)
        assert df["a"].tolist() == [3.0, 6.0]
    assert len(grown_runs) == 3
    assert len(scaled_runs) == 1


def _sort_and_extend(v):
    v.sort(reverse=True)
    v.append(-1)
    return len(v)


def _count_and_drop(d):
    d["n"] = d.get("n", 0) + 7
    del d["gone"]
    return len(d)


@pytest.mark.parametrize(
    "change,make",
    [
        pytest.param(_sort_and_extend, lambda: [3, 1, 2], id="list"),
        pytest.param(_count_and_drop, lambda: {"n": 1, "gone": 2, "k": "v"}, id="dict"),
    ],
)
def test_a_list_or_dict_is_stored_from_the_first_call(disk_cash, change, make):
    runs = []

    @disk_cash.cache
    def work(v):
        runs.append(1)
        return change(v)

    work(make())
    v = make()
    result = work(v)
    assert len(runs) == 1
    expected = make()
    assert result == change(expected)
    assert type(v) is type(expected) and v == expected


def test_a_frame_is_stored_from_the_first_call(disk_cash):
    runs = []

    @disk_cash.cache
    def work(df):
        runs.append(1)
        df["a"] *= 2
        df.loc[0, "b"] = "z"
        return float(df["a"].sum())

    def make():
        return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "w"]})

    work(make())
    df = make()
    assert work(df) == 12.0
    assert len(runs) == 1
    expected = make()
    expected["a"] *= 2
    expected.loc[0, "b"] = "z"
    assert df.equals(expected)


SCRIPT = textwrap.dedent("""
    import numpy as np
    import cash

    @cash.cache
    def clip_in_place(a):
        np.clip(a, 0, 1, out=a)
        return float(a.sum())

    a = np.array([-1.0, 0.5, 2.0])
    result = clip_in_place(a)
    print(result, a.tolist(), clip_in_place.cache_info()["hits"])
""")


def test_a_new_process_serves_the_first_call_and_writes_the_state_back(tmp_path):
    (tmp_path / "job.py").write_text(SCRIPT)
    first = run_python("job.py", cwd=tmp_path).stdout.split()
    second = run_python("job.py", cwd=tmp_path).stdout.split()
    assert first == ["1.5", "[0.0,", "0.5,", "1.0]", "0"]
    assert second == ["1.5", "[0.0,", "0.5,", "1.0]", "1"], "the first process's call was not served"


class Box:
    pass


def _observed(func, *args):
    """The IMPURE-OBSERVED-EFFECTS warnings calling *func* on each of
    *args* in turn shows, and the first line of each it logs."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for arg in args:
            func(arg)
    shown = [str(w.message) for w in caught if getattr(w.message, "code", None) == "IMPURE-OBSERVED-EFFECTS"]
    logged = [entry["message"] for entry in func.cache_info()["warnings"] if entry["code"] == "IMPURE-OBSERVED-EFFECTS"]
    return shown, logged


def test_the_warning_says_what_a_hit_does_when_it_writes_the_state_back(full_warning_text, disk_cash):
    """One plain line where a hit writes the argument back; the hazard as
    before where it cannot."""

    @disk_cash.cache
    def clip_in_place(a):
        return _clip(a)

    @disk_cash.cache
    def fill(box):
        vars(box).update(n=1)
        return 1

    shown, logged = _observed(clip_in_place, np.array([-1.0, 0.5, 2.0]), np.array([-1.0, 0.5, 2.0]))
    line = (
        "[IMPURE-OBSERVED-EFFECTS] clip_in_place changed 'a' in place; cash stores the state it leaves 'a' in, "
        "and a hit writes it back."
    )
    assert [text.splitlines()[0] for text in shown] == [line], shown
    assert [text.splitlines()[0] for text in logged] == [line], logged
    assert "will not repeat" not in shown[0] and "WITHOUT" not in shown[0] and "Fix:" not in shown[0]

    shown, logged = _observed(fill, Box(), Box())
    assert len(shown) == len(logged) == 1, shown
    assert "Every cache HIT from here on returns the stored value WITHOUT repeating them" in logged[0]
    assert "the call changed 'box' in place -- the result was not stored, so this call runs every time" in logged[0]


def test_a_change_written_in_the_body_is_said_once(full_warning_text, disk_cash):
    """``v.sort()`` in the body is read from the source too. Where a hit
    writes the list back, the one line is the only warning; a list of lists,
    which a hit cannot write back, keeps the source finding."""

    @disk_cash.cache
    def sort_values(v):
        v.sort()
        return len(v)

    @disk_cash.cache
    def sort_rows(rows):
        rows.sort()
        return len(rows)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            v = [3, 1, 2]
            sort_values(v)
            assert v == [1, 2, 3]
    shown = [(w.message.code, str(w.message).splitlines()[0]) for w in caught if hasattr(w.message, "code")]
    assert shown == [
        (
            "IMPURE-OBSERVED-EFFECTS",
            "[IMPURE-OBSERVED-EFFECTS] sort_values changed 'v' in place; cash stores the state it leaves 'v' in, "
            "and a hit writes it back.",
        )
    ], shown
    assert sort_values.cache_info()["hits"] == 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            sort_rows([[2], [1]])
    texts = [str(w.message) for w in caught if getattr(w.message, "code", None) == "IMPURE-SIDE-EFFECTS"]
    assert len(texts) == 1, [str(w.message) for w in caught]
    assert "changes the argument 'rows' in place; a cache hit would not make that change" in texts[0]
    assert sort_rows.cache_info()["hits"] == 0


def test_a_call_not_stored_after_all_does_not_say_a_hit_writes_back(disk_cash):
    """``return X[:2]`` shares the argument's memory, so the call is not
    stored: no line may say a hit writes the state back. The same change
    returning a copy is stored and says so."""

    @disk_cash.cache
    def head_view(X):
        X *= 2
        return X[:2]

    @disk_cash.cache
    def head_copy(X):
        X *= 2
        return X[:2].copy()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        head_view(np.arange(4.0))
        head_copy(np.arange(4.0))
    said = [str(w.message) for w in caught if "a hit writes it back" in str(w.message)]
    assert len(said) == 1 and "head_copy changed 'X'" in said[0], said


def test_a_function_that_starts_changing_its_argument_later_is_stored_from_the_call_after(disk_cash):
    """After a miss that changed nothing, nothing of the arguments is taken
    before the body: the first change is not stored, the next one is."""
    runs = []

    @disk_cash.cache
    def maybe_clip(a, clip):
        runs.append(1)
        if clip:
            np.clip(a, 0, 1, out=a)
        return float(a.sum())

    maybe_clip(np.array([-1.0, 2.0]), False)
    for _ in range(3):
        a = np.array([-1.0, 2.0])
        assert maybe_clip(a, True) == 1.0
        assert a.tolist() == [0.0, 1.0]
    assert len(runs) == 3

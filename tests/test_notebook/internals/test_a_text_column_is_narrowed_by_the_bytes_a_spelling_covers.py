"""An Arrow text column is read for ids only where a value could spell
one, and that finds what reading all of it found.

The identity check looks in every Arrow text column of a statement's new
frame for values that spell an ``id()`` anywhere in them: a run of 7 or
more ASCII digits, or ``0x`` and hexadecimal digits. Arrow measured, masked
and filtered every value first: 0.13 s for p18's column of 5.6 million
subject ids (``S011-00001``, each of a length that fits), after each
statement making a new frame of lab rows. A spelling is 7 bytes or more of
digits, ``x`` and hexadecimal letters, so the values are now picked from
the offsets and the few bytes every spelling in them would cover, through
NumPy (`_could_spell`), and Arrow's patterns run on those alone. These
tests pin that the numbers found are the same, for any text, slice,
chunking, missing value and offset width, and the same as `_spelled_in`
reads in each value.
"""

from __future__ import annotations

import random

import pytest

from cash.notebook.statement import identity_holders
from cash.notebook.statement.identity_holders import identity_held_reason

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")
pd = pytest.importorskip("pandas")


def _read(chunked, narrowed: bool, monkeypatch) -> list:
    parts: list = []
    with monkeypatch.context() as patch:
        if not narrowed:
            patch.setattr(identity_holders, "_could_spell", lambda pyarrow, chunked: None)
        identity_holders._spelled_numbers(pa, pc, chunked, parts)
    return [part.tolist() if hasattr(part, "tolist") else part for part in parts]


def _numbers(parts: list) -> set:
    found: set = set()
    for part in parts:
        found.update(part if isinstance(part, list) else [part])
    return found


def _texts(rng: random.Random) -> list:
    letters = "0123456789xXabcdefAFg-é \n"
    texts: list = []
    for _ in range(rng.randint(0, 300)):
        pick = rng.random()
        if pick < 0.1:
            texts.append(None)
        elif pick < 0.3:
            texts.append(str(rng.randint(10**5, 10**18 - 1)))
        elif pick < 0.4:
            texts.append(
                "0" + rng.choice("xX") + "".join(rng.choice("0123456789abcdefAFg") for _ in range(rng.randint(3, 17)))
            )
        else:
            texts.append("".join(rng.choice(letters) for _ in range(rng.randint(0, 20))))
        if texts and texts[-1] is not None and rng.random() < 0.2:
            # Inside a longer text, now and then a long one.
            texts[-1] = (
                rng.choice(["k", "run-", "S-", "", "x" * rng.randint(0, 40)])
                + texts[-1]
                + rng.choice(["", "-x", ";", "zz", "\n", " " * rng.randint(0, 30)])
            )
    return texts


@pytest.mark.parametrize("seed", range(40))
def test_the_numbers_found_are_those_reading_all_of_it_finds(seed, monkeypatch):
    rng = random.Random(seed)
    for _ in range(10):
        kind = rng.choice([pa.string(), pa.large_string()])
        column = pa.array(_texts(rng), type=kind)
        if len(column) > 4 and rng.random() < 0.5:
            column = column.slice(rng.randint(0, 3), rng.randint(0, len(column) - 4))
        if len(column) > 3 and rng.random() < 0.5:
            cut = rng.randint(1, len(column) - 1)
            column = pa.chunked_array([column.slice(0, cut), column.slice(cut)], type=kind)
        narrowed = _read(column, True, monkeypatch)
        assert narrowed == _read(column, False, monkeypatch)
        expected: set = set()
        for text in column.to_pylist():
            if text is not None:
                expected |= identity_holders._spelled_in(text)
        assert _numbers(narrowed) == expected


@pytest.mark.parametrize("kind", [pa.string(), pa.large_string()])
def test_empty_and_missing_values_at_the_end_are_read(kind, monkeypatch):
    for texts in (
        [],
        [None],
        [""],
        ["", ""],
        ["12345678", ""],
        ["S011-00001", None, ""],
        ["0x7f00aa12bc", "", None],
        ["", "k12345678"],
        ["a" * 30 + "0x7f00aa12bc", ""],
    ):
        column = pa.array(texts, type=kind)
        assert _read(column, True, monkeypatch) == _read(column, False, monkeypatch)


def test_values_that_cannot_spell_a_number_are_not_matched(monkeypatch):
    seen: list[int] = []
    real = pc.ascii_is_decimal

    def counting(values, *args, **kwargs):
        seen.append(len(values))
        return real(values, *args, **kwargs)

    monkeypatch.setattr(pc, "ascii_is_decimal", counting)
    column = pa.array([f"S{i % 97:03d}-{i:05d}" for i in range(20_000)] + ["12345678"], type=pa.large_string())
    # Every one of 20,000 ids ruled out by its bytes alone: only the number
    # is matched.
    parts: list = []
    identity_holders._spelled_numbers(pa, pc, column, parts)
    assert [part.tolist() for part in parts] == [[12345678]]
    assert seen == [1]


@pytest.mark.parametrize("spelled", ["{d}", "S-{d}", "run {h} done", "<Q object at {h}>" + " pad" * 10])
def test_an_id_spelled_in_a_long_column_of_ids_is_still_found(spelled):
    held = object()
    ids = [f"S{i % 97:03d}-{i:05d}" for i in range(50_000)]
    ids[31_337] = spelled.format(d=id(held), h=hex(id(held)))
    frame = pd.DataFrame({"subject_id": pd.array(ids, dtype="string[pyarrow]"), "n": range(50_000)})
    identity_holders.forget_flat_reads()
    try:
        reason = identity_held_reason("frame", frame, {"frame": frame, "held": held})
    finally:
        identity_holders.forget_flat_reads()
    assert reason is not None
    assert "id()" in str(reason)

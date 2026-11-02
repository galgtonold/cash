"""Only the values of a text column long enough to spell an address are
matched against the number patterns.

The identity check looks in every Arrow text column of a statement's new
frame for numbers that could be an ``id()``: values of 7 to 18 bytes, all
digits or starting ``0x``. Both patterns ran over every value of the
column, the prefix as a match ignoring case (a regular expression in
Arrow): after p18's ``chg = labs.merge(...)``, most of the check's time,
for nine columns of country names, visit codes and the like.
"""

from __future__ import annotations

import pytest

from cash.notebook.statement import identity_holders

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")


@pytest.fixture
def matched(monkeypatch):
    """The values each pattern was matched against."""
    seen: list[int] = []
    for name in ("ascii_is_decimal", "starts_with"):
        real = getattr(pc, name)

        def counting(values, *args, _real=real, **kwargs):
            seen.append(len(values))
            return _real(values, *args, **kwargs)

        monkeypatch.setattr(pc, name, counting)
    return seen


def test_short_values_are_not_matched(matched):
    short = ["V1", "V10", "Germany", "Australia"] * 5_000  # two of them fit
    texts = [*short, "12345678", "0x7f00aa12bc", "0X7F00AA12BD", "0xzz000000"]
    s = pd.Series(texts, dtype="str" if pd.__version__ >= "3" else "string[pyarrow]")
    identity_holders.forget_flat_reads()
    parts = identity_holders._arrow_text_spelling_numbers(s)
    assert parts is not None
    assert sorted(parts[0].tolist()) == [12345678]
    assert sorted(parts[1:]) == [0x7F00AA12BC, 0x7F00AA12BD]
    # Positive control: the patterns ran, on the 10,004 values that fit.
    assert matched and max(matched) == 10_004
    assert sum(matched) <= 3 * 10_004

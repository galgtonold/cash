"""Only the values of a text column long enough to spell an address are
matched against the number patterns.

The identity check looks in every Arrow text column of a statement's new
frame for numbers that could be an ``id()``: a run of 7 digits or ``0x``
and hexadecimal digits, anywhere in a value. Both patterns ran over every
value of the column, the prefix as a match ignoring case (a regular expression in
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
    for name in ("ascii_is_decimal", "match_substring_regex"):
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
    found = {int(n) for part in parts for n in (part.tolist() if hasattr(part, "tolist") else [part])}
    assert found == {12345678, 0x7F00AA12BC, 0x7F00AA12BD}
    # Positive control: the patterns ran, on the 3 values whose bytes could
    # spell a number (``Germany``, ``Australia`` and ``0xzz000000`` are
    # long enough, but a letter that is no hexadecimal digit sits where any
    # spelling in them would be: `_could_spell`).
    assert matched and max(matched) == 3
    assert sum(matched) <= 2 * 3

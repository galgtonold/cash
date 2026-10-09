"""What a statement read from a stream is a new value on every run.

``text = fh.read()`` runs every time (a hit would not move the file), but the
lineage it gave ``text`` came from its code and the handle, the same each run.
After a second read, which returns ``''``, ``n = text.count('a')`` was served
the count of the first read's text, in the namespace too: 1973863 where a plain
kernel has 0 (found replaying a student's notebook of the JuNE dataset,
'expert_7' step 22).
"""

from __future__ import annotations

import warnings

import pytest

from tests._cell_driver import run_cash_cell

# Slow enough to be worth storing.
COUNT = "n = sum(1 for ch in text if ch == 'a')\nprint(n)"
READS = {"read": "text = fh.read()", "readline": "text = fh.readline()"}


@pytest.mark.parametrize("read", sorted(READS))
def test_a_second_read_is_not_served_the_first_ones_result(cash_magics, tmp_path, read):
    data = tmp_path / "probe_data.txt"
    data.write_text("a" * 1_000_000 + "\n" + "a" * 1_000_000, encoding="utf-8")
    ns = cash_magics.shell.user_ns
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_cash_cell(cash_magics, f"fh = open(r'{data}', encoding='utf-8')")
        run_cash_cell(cash_magics, READS[read])
        run_cash_cell(cash_magics, COUNT)
        first = ns["n"]
        ns["fh"].read()  # read the rest: the next reader finds the end
        run_cash_cell(cash_magics, READS[read])
        run_cash_cell(cash_magics, COUNT)
    assert first > 0
    assert ns["n"] == ns["text"].count("a"), f"count of an old text: {ns['n']}"
    if read == "read":
        assert ns["n"] == 0

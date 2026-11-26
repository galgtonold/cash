"""What a magic prints is plain text: markers such as ``[OK]``, ``[!]`` and
``[Error]``, no emoji.

``%cash_provenance`` headed its answer with a clipboard and marked each
timeline line with a wrench or a parcel, where every other magic prints
words. One style reads the same in a terminal, a log file and a notebook,
and needs no fallback on a console that cannot encode the symbols.
"""

from __future__ import annotations

import re

import pytest

from tests._cell_driver import run_cash_cell

#: Pictographs, dingbats and their variation selector. Arrows and the
#: box-drawing lines of the dependency graph are text and may stay.
_EMOJI = re.compile("[\U0001f000-\U0001ffff☀-➿⬀-⯿⌀-⏿️]")

#: Each magic with the lines that make it print every form of its output.
MAGIC_LINES = {
    "cash_on": ["", "ttl=60", "ttl=soon", "nonsense"],
    "cash_off": ["", "nonsense"],
    "cash_debug": ["on", "off", "", "nonsense"],
    "cash_persist": ["on", "off", "", "nonsense"],
    "cash_help": ["", "badge", "provenance", "nonsense"],
    "cash_badge": ["print", "off", "", "nonsense"],
    "cash_status": ["", "json", "nonsense"],
    "cash_stats": ["", "reset", "nonsense"],
    "cash_provenance": ["", "--all", "total", "total --graph --time", "missing", "total --nonsense", "--clear"],
}


def test_every_magic_is_covered(cash_magics):
    assert set(MAGIC_LINES) == set(cash_magics.magics["line"]), "a new magic: add the lines that make it print"


@pytest.mark.parametrize("magic", sorted(MAGIC_LINES))
def test_a_magic_prints_no_emoji(cash_magics, capsys, magic):
    run_cash_cell(cash_magics, "data = list(range(5))\ntotal = sum(data)")
    run_cash_cell(cash_magics, "data = list(range(5))\ntotal = sum(data)")
    capsys.readouterr()

    printed = []
    for line in MAGIC_LINES[magic]:
        getattr(cash_magics, magic)(line)
        captured = capsys.readouterr()
        printed.append(captured.out + captured.err)

    text = "\n".join(printed)
    assert text.strip(), "control: the magic printed something"
    assert not _EMOJI.findall(text), text


def test_the_provenance_of_a_variable_is_headed_and_timed_in_words(cash_magics, capsys):
    provenance = cash_magics.session.provenance
    provenance.record("x", "x = 1", [], status="computed", duration_ms=2.0)
    provenance.record("x", "x = 1", [], status="restored", duration_ms=1.0)
    provenance.record("x", "x = 1", [], status="skipped")
    provenance.record("x", "x = 1", [], status="another")

    cash_magics.cash_provenance("x --time")

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Provenance for 'x':"
    timeline = [line.split(None, 1)[1] for line in lines[lines.index("  Timeline:") + 1 :] if line.strip()]
    assert timeline == ["EXECUTED (2.0ms)", "CACHED (1.0ms)", "SKIPPED (0.0ms)", "another (0.0ms)"]

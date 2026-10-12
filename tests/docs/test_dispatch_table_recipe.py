"""The "Dispatch tables and strategy registries" recipe must be true.

``docs/decorator-limitations.md`` says a module-level table of functions is
keyed by every function in it (editing an unused strategy recomputes the used
one), and that passing the function itself through a thin uncached front keys
each strategy on its own (editing an unused strategy keeps the used one's
entries). Both are claims about what happens across an edit, between two
processes, which the docs harness cannot model, and the second fence needs its
functions in a real module, so the second fence is ``test:skip`` there and run
here as a script.

The fences are read out of the page, so the code under test is what a reader
sees.
"""

from __future__ import annotations

import pathlib
import re

from tests._scripts import run_python

REPO = pathlib.Path(__file__).resolve().parents[2]
PAGE = REPO / "docs" / "decorator-limitations.md"

_FENCE = re.compile(r"^```python\s*$(.*?)^```\s*$", re.MULTILINE | re.DOTALL)

_REPORT = """
print("table", signals_from_table.cache_info()["hits"], signals_from_table.cache_info()["misses"])
print("front", _signals.cache_info()["hits"], _signals.cache_info()["misses"])
"""


def _recipe() -> str:
    text = PAGE.read_text(encoding="utf-8")
    start = text.index("## Dispatch tables and strategy registries")
    end = text.index("\n## ", start + 10)
    fences = [m.group(1) for m in _FENCE.finditer(text[start:end])]
    assert len(fences) == 2, f"expected 2 python fences in the recipe, found {len(fences)}; update this test with it"
    return fences[0] + "\n" + fences[1] + _REPORT


def _counts(stdout: str) -> dict[str, tuple[int, int]]:
    out = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] in {"table", "front"}:
            out[parts[0]] = (int(parts[1]), int(parts[2]))
    return out


def test_the_table_is_keyed_by_every_strategy_and_the_front_by_its_own(tmp_path):
    script = tmp_path / "recipe.py"
    source = _recipe()
    script.write_text(source, encoding="utf-8")

    first = _counts(run_python(script.name, cwd=tmp_path).stdout)
    # (hits, misses): the second identical call hits in both forms.
    assert first == {"table": (1, 1), "front": (1, 2)}, first

    again = _counts(run_python(script.name, cwd=tmp_path).stdout)
    assert again == {"table": (2, 0), "front": (3, 0)}, f"a fresh process did not hit: {again}"

    # Edit only the strategy no call above uses with signals_from_table.
    edited = source.replace("return [mean - p for p in prices]", "return [p - mean for p in prices]")
    assert edited != source, "the recipe's mean_reversion changed; update this test with it"
    script.write_text(edited, encoding="utf-8")
    after = _counts(run_python(script.name, cwd=tmp_path).stdout)
    # The table form recomputes "momentum" although momentum did not change;
    # the front keeps the momentum entry and recomputes only mean_reversion.
    assert after == {"table": (1, 1), "front": (2, 1)}, after

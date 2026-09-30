"""``cash clear --entry "" PATH`` refuses instead of clearing the whole cache.

``cash clear --entry "$ID" ./cache`` with ``$ID`` unset passes an empty id.
The empty string is falsy, so it fell through to the plain "clear this path"
branch and removed every entry. A selector that selects nothing must exit 2
and leave the cache alone; the same holds for ``--function ""``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

SCRIPT = """
from cash import Cash
app = Cash(cache_dir={cache!r})

@app.cache
def f(x):
    return x

f(1); f(2)
"""


@pytest.mark.parametrize("flag", ["--entry", "--function"])
@pytest.mark.parametrize("value", ["", "  "])
def test_an_empty_selector_clears_nothing(tmp_path, flag, value):
    cache = tmp_path / "cache"
    script = tmp_path / "mk.py"
    script.write_text(SCRIPT.format(cache=str(cache)), encoding="utf-8")
    subprocess.run([sys.executable, str(script)], check=True, cwd=tmp_path)
    before = sorted(p.name for p in cache.iterdir())
    assert before, "the script wrote no cache to clear"

    proc = subprocess.run(
        [sys.executable, "-m", "cash", "clear", flag, value, str(cache)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "nothing was cleared" in proc.stdout, proc.stdout
    assert cache.is_dir(), proc.stdout
    assert sorted(p.name for p in cache.iterdir()) == before

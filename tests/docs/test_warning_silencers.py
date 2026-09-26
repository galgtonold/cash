"""Each impurity code's **Silencing it** line on `docs/warnings.md` is true.

The waivers do not all reach every code: `assume_safe=True` does not silence
IMPURE-SCOPE-MUTATION, a `# @cash:assume-safe` line cannot reach
CACHE-RESULT-SHARED, and KEY-OPAQUE-CALLABLE answers to neither. A page that
promised both waivers everywhere would send readers to one that does nothing,
so every claim is run here: the code fires bare, and each waiver the page
names silences it while each one it rules out does not.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

from cash import Cash

PAGE = Path(__file__).resolve().parents[2] / "docs" / "warnings.md"

#: code -> (module source, call). ``{W}`` is where the line waiver goes and
#: ``{DEC}`` the decorator's arguments.
CASES: dict[str, tuple[str, str]] = {
    "CACHE-RESULT-SHARED": (
        """
        @c.cache{DEC}
        def f(a):
            return a{W}
        """,
        "f([1, 2, 3])",
    ),
    "IMPURE-OBSERVED-EFFECTS": (
        """
        import zipfile

        @c.cache{DEC}
        def f(target):
            with zipfile.ZipFile(target, "w") as archive:{W}
                names = archive.namelist()
            return str(names)
        """,
        "f(str(tmp / 'out.zip'))",
    ),
    "IMPURE-SCOPE-MUTATION": (
        """
        class Counter:
            def __init__(self):
                self.n = 0

            def bump(self):
                self.n += 1  # @cash:assume-safe

        COUNTER = Counter()

        @c.cache{DEC}
        def f(x):
            y = x * 2
            COUNTER.bump(){W}
            return y
        """,
        "f(1); f(2)",
    ),
    "IMPURE-SIDE-EFFECTS": (
        """
        import os

        @c.cache{DEC}
        def f(x):
            os.remove("/nonexistent/cash-doc-test") if x < 0 else None{W}
            return x
        """,
        "f(1)",
    ),
    "KEY-AMBIENT-READ": (
        """
        from datetime import datetime

        @c.cache{DEC}
        def f(x):
            t = datetime.now(){W}
            return x, t.year
        """,
        "f(1)",
    ),
    "KEY-DYNAMIC-DEPENDENCY": (
        """
        import math

        class Carrier:
            def run(self, name):
                return getattr(math, name)(1.0){W}

        @c.cache{DEC}
        def f(obj):
            return obj.run("sqrt")
        """,
        "f(Carrier())",
    ),
    "KEY-FROZEN-MUTATED": (
        """
        @c.cache(frozen=True{FDEC})
        def f(x):
            return [x]{W}

        @c.cache
        def g(v):
            return len(v)
        """,
        "r = f(1)\nfor i in range(10):\n    r.append(i)\n    g(r)",
    ),
    "KEY-NETWORK-READ": (
        """
        import urllib.request

        @c.cache{DEC}
        def f(u):
            if u == "never":
                return urllib.request.urlopen(u).read(){W}
            return 1
        """,
        "f('x')",
    ),
    "KEY-OPAQUE-CALLABLE": (
        """
        import re

        P = re.compile("a").match

        @c.cache{DEC}
        def f(x, h=P):
            return bool(h(x)){W}
        """,
        "f('a')",
    ),
    "KEY-UNHASHABLE-GLOBAL": (
        """
        import threading

        LOCK = threading.Lock()

        @c.cache{DEC}
        def f(x):
            return (x, LOCK.locked()){W}
        """,
        "f(1)",
    ),
}


def _silencing_line(code: str) -> str:
    text = PAGE.read_text("utf-8")
    start = text.index(f"### {code} {{#")
    end = text.find("\n### ", start + 1)
    section = text[start : end if end != -1 else len(text)]
    m = re.search(r"\*\*Silencing it\.\*\*(.*?)(?:\n\n|\Z)", section, re.S)
    assert m, f"{code}: no **Silencing it.** line"
    return " ".join(m.group(1).split())


def _documented(code: str) -> tuple[bool, bool]:
    """(line waiver silences it, assume_safe=True silences it), as the page says."""
    line = _silencing_line(code)
    return line.startswith("Per line:"), "Whole function: `@cash.cache(assume_safe=True)`" in line


def _fires(tmp_path: Path, code: str, variant: str) -> bool:
    body, call = CASES[code]
    body = textwrap.dedent(body)
    body = body.replace("{W}", "  # @cash:assume-safe" if variant == "line" else "")
    body = body.replace("{DEC}", "(assume_safe=True)" if variant == "func" else "")
    body = body.replace("{FDEC}", ", assume_safe=True" if variant == "func" else "")
    work = tmp_path / variant
    work.mkdir()
    name = f"silencer_{code.lower().replace('-', '_')}_{variant}"
    path = work / f"{name}.py"
    path.write_text(body, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.c = Cash(cache_dir=str(work / ".cash"), register_magic=False)
    sys.modules[name] = module
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            spec.loader.exec_module(module)
            exec(call, {**vars(module), "tmp": work})
    finally:
        sys.modules.pop(name, None)
    return any(getattr(w.message, "code", None) == code for w in record)


@pytest.mark.parametrize("code", sorted(CASES))
def test_the_code_fires_without_a_waiver(tmp_path, code):
    """The positive control: without it every 'silenced' below is vacuous."""
    assert _fires(tmp_path, code, "plain")


@pytest.mark.parametrize("code", sorted(CASES))
def test_the_line_waiver_does_what_the_page_says(tmp_path, code):
    per_line, _ = _documented(code)
    assert _fires(tmp_path, code, "line") is not per_line, _silencing_line(code)


@pytest.mark.parametrize("code", sorted(CASES))
def test_assume_safe_does_what_the_page_says(tmp_path, code):
    _, whole = _documented(code)
    assert _fires(tmp_path, code, "func") is not whole, _silencing_line(code)


def test_every_impurity_code_is_covered():
    """A new CashImpurityWarning code needs a case here and a line on the page."""
    text = PAGE.read_text("utf-8")
    impurity = set(re.findall(r"^### ([A-Z-]+) \{#[a-z-]+\}\n\n.*CashImpurityWarning", text, re.M))
    assert impurity == set(CASES), impurity ^ set(CASES)

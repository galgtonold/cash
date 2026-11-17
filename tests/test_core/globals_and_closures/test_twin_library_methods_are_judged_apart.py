"""Two libraries' methods of the same text are judged each on its own.

Two code objects compare equal when their bytecode, names, constants and
lines do, whatever module they come from. What the recorder keeps about a
library's code (whether a method may read a class setting off ``self``, the
chains it reads, the names it loads off its first argument) was kept in
weak dictionaries keyed by that equality, which also hashed the code's
bytecode again at every lookup: a ``pvlib`` call starting 160 of pandas'
methods spent about 1 ms a miss on those hashes alone. Keyed by equality,
``twin_a.Model.compute`` (no class setting anywhere) gave its verdict to
``twin_b.Model.compute``, whose class does hold ``scale = 1``: the method
was then taken to read no setting and never watched again, and a call made
after ``twin_b.Model.scale`` changed was served the old result. Each code
object is now judged and remembered by identity.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from cash import Cash

pytestmark = [
    pytest.mark.core,
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashCacheIneffectiveWarning"),
    pytest.mark.skipif(
        sys.version_info < (3, 12), reason="a body's reads are recorded with sys.monitoring, Python 3.12 or newer"
    ),
]

# ``compute`` sits on the same lines with the same text in both modules,
# so the two code objects compare equal.
TWIN_A = """
class Model:
    def compute(self):
        return self.x * self.scale

    def __init__(self, x):
        self.x = x
        self.scale = 2
"""

TWIN_B = """
class Model:
    def compute(self):
        return self.x * self.scale

    def __init__(self, x):
        self.x = x

    scale = 1
"""


def _install(root):
    site = root / "venv" / "lib" / "site-packages"
    package = site / "cash_ut_twins"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "twin_a.py").write_text(TWIN_A, encoding="utf-8")
    (package / "twin_b.py").write_text(TWIN_B, encoding="utf-8")
    info = site / "cash_ut_twins-1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: cash-ut-twins\nVersion: 1.0\n", encoding="utf-8")
    (info / "top_level.txt").write_text("cash_ut_twins\n", encoding="utf-8")
    (info / "RECORD").write_text(
        "cash_ut_twins/__init__.py,,\ncash_ut_twins/twin_a.py,,\ncash_ut_twins/twin_b.py,,\n"
        "cash_ut_twins-1.0.dist-info/METADATA,,\n",
        encoding="utf-8",
    )
    return site


@pytest.fixture
def twins(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(_install(tmp_path)))
    for name in ("cash_ut_twins", "cash_ut_twins.twin_a", "cash_ut_twins.twin_b"):
        sys.modules.pop(name, None)
    a = importlib.import_module("cash_ut_twins.twin_a")
    b = importlib.import_module("cash_ut_twins.twin_b")
    yield a, b
    for name in ("cash_ut_twins", "cash_ut_twins.twin_a", "cash_ut_twins.twin_b"):
        sys.modules.pop(name, None)


def test_a_twins_verdict_does_not_hide_a_class_setting(twins):
    a, b = twins
    assert a.Model.compute.__code__ == b.Model.compute.__code__
    assert a.Model.compute.__code__ is not b.Model.compute.__code__
    c = Cash(backend="memory", register_magic=False)

    @c.cache
    def via_a(x):
        return a.Model(x).compute()

    @c.cache
    def via_b(x):
        return b.Model(x).compute()

    assert via_a(7) == 14
    assert via_a(7) == 14
    for scale in (1, 5, 2, 5, 1):
        b.Model.scale = scale
        assert via_b(7) == 7 * scale, scale
        assert via_b(7) == 7 * scale, scale

"""KEY-AMBIENT-READ recognises a clock read however the module was imported.

Round 19 (r19s3, r19s4): ``import datetime as _dt; _dt.datetime.now()``,
``from datetime import datetime as DateTime``, ``import time as _time``,
``import os as _os``, ``pd.Timestamp.now()`` and ``pd.to_datetime("today")``
froze a timestamp into every later result with no warning, while the
canonical spellings warned.
"""
from __future__ import annotations

import importlib
import sys
import textwrap
import time
import warnings

import pytest

from cash import Cash

pytestmark = [pytest.mark.core]

HEADER = """\
import datetime
import datetime as _dt
import os as _os
import time as _time
from datetime import date as Day, datetime as DateTime
from time import time as now_ts
"""

SPELLINGS = {
    "module-alias": "_dt.datetime.now()",
    "module-alias-date": "_dt.date.today()",
    "class-alias": "DateTime.now()",
    "class-alias-utcnow": "DateTime.utcnow()",
    "date-alias": "Day.today()",
    "time-alias": "_time.time()",
    "function-alias": "now_ts()",
    "os-alias-getcwd": "_os.getcwd()",
    "os-alias-environ": '_os.environ.get("HOME")',
    "canonical": "datetime.datetime.now()",
}

PANDAS_SPELLINGS = {
    "pd-now": "pd.Timestamp.now()",
    "pd-today": "pd.Timestamp.today()",
    "pd-to-datetime-today": 'pd.to_datetime("today")',
    "pd-timestamp-now-arg": 'pd.Timestamp("now")',
}


def _load(tmp_path, monkeypatch, body, extra=""):
    name = f"ambient_{time.monotonic_ns()}"
    (tmp_path / f"{name}.py").write_text(
        HEADER + extra + textwrap.dedent(f"""

        def stamp(n):
            return {body}
        """), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        return importlib.import_module(name)
    finally:
        monkeypatch.setitem(sys.modules, name, sys.modules[name])


def _ambient_warnings(tmp_path, fn):
    c = Cash(cache_dir=str(tmp_path / "cache"))
    cached = c.cache(fn)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cached(1)
    return [w for w in rec if "KEY-AMBIENT-READ" in str(w.message)]


@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_an_aliased_ambient_read_warns(tmp_path, monkeypatch, spelling):
    mod = _load(tmp_path, monkeypatch, SPELLINGS[spelling])
    assert _ambient_warnings(tmp_path, mod.stamp), f"{SPELLINGS[spelling]} froze silently"


@pytest.mark.parametrize("spelling", sorted(PANDAS_SPELLINGS))
def test_a_pandas_clock_read_warns(tmp_path, monkeypatch, spelling):
    pytest.importorskip("pandas")
    mod = _load(tmp_path, monkeypatch, PANDAS_SPELLINGS[spelling], extra="import pandas as pd\n")
    assert _ambient_warnings(tmp_path, mod.stamp), f"{PANDAS_SPELLINGS[spelling]} froze silently"


@pytest.mark.parametrize("body, extra", [
    ("Clock().now()", "class Clock:\n    def now(self):\n        return 1\n"),
    ("Stamp.now()", "class Stamp:\n    @staticmethod\n    def now():\n        return 1\n"),
    ("pd.to_datetime('2024-01-01')", "import pandas as pd\n"),
])
def test_a_now_that_is_not_the_clock_does_not_warn(tmp_path, monkeypatch, body, extra):
    """Control: resolution goes through what the name IS -- the user's own
    `now`, and a date parsed from a fixed string, are not ambient reads."""
    if "pandas" in extra:
        pytest.importorskip("pandas")
    mod = _load(tmp_path, monkeypatch, body, extra=extra)
    assert not _ambient_warnings(tmp_path, mod.stamp)

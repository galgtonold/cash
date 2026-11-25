"""However a program writes into a frame, a cached function sees it.

The frame memo answers for a frame it has hashed without reading it again,
for as long as nothing wrote to it. What "nothing wrote to it" rests on is
a list of routes (``cash._pandas_compat.WRITE_ROUTES``): each way pandas
lets a frame's memory be written, through its own methods, through a view
made writable again, through a handle to the array it keeps, through the
memory it hands to a callback. A route that is missing from the list is a
stale result, and a new pandas can add one.

This is the probe of that list. It runs every route of the start-up check,
and the less everyday ones the check leaves out (ctypes, the array
interface, the interchange protocol, another thread, ``inplace=`` methods),
through ``@cash.cache`` in a fresh interpreter: the function is called,
the frame is written through the route, the function is called again and
must answer for the frame as it now is.

It runs with the version gate lifted, so that it tells something about a
pandas cash has not been tested with: the nightly workflow runs it against
the newest pandas pre-release. When it passes there,
``cash._pandas_compat.NEWEST_TESTED`` can be raised.
"""

from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from cash import _pandas_compat
from tests._scripts import run_python

pytestmark = pytest.mark.skipif(
    int(pd.__version__.split(".")[0]) < 3, reason="copy-on-write, which the frame memo needs, is pandas 3's"
)

_WRITABLE = "v.flags.writeable = True"
#: Routes beside those of the start-up check, in its form:
#: ``(name, frame kind, statements over pd, np and df)``.
MORE_ROUTES = (
    ("asarray of an index, as handed out", "num", "np.asarray(df.index)[0] = 99"),
    ("asarray of the columns", "numcols", "np.asarray(df.columns)[0] = 99"),
    (
        "ctypes over a column's memory",
        "num",
        "import ctypes\n    ctypes.c_double.from_address(df['a'].values.ctypes.data).value = 9.0",
    ),
    (
        "ctypes over the frame's memory",
        "num",
        "import ctypes\n    ctypes.c_double.from_address(df.to_numpy().ctypes.data).value = 9.0",
    ),
    (
        "the array interface",
        "num",
        "class Borrowed:\n        pass\n"
        "    b = Borrowed()\n"
        "    spec = dict(df['a'].values.__array_interface__)\n"
        "    spec['data'] = (spec['data'][0], False)\n"
        "    b.__array_interface__ = spec\n"
        "    b.keep = df\n"
        "    np.asarray(b)[0] = 9.0",
    ),
    (
        "the interchange protocol",
        "num",
        "import ctypes\n"
        "    buffer = df.__dataframe__().get_column(0).get_buffers()['data'][0]\n"
        "    ctypes.c_double.from_address(buffer.ptr).value = 9.0",
    ),
    ("a frame over np.array without a copy", "num", f"v = np.array(df, copy=False); {_WRITABLE}; v[0, 0] = 9.0"),
    ("to_numpy without a copy", "num", f"v = df['a'].to_numpy(copy=False); {_WRITABLE}; v[0] = 9.0"),
    ("the transposed values", "num", f"v = df.T.values; {_WRITABLE}; v[0, 0] = 9.0"),
    ("a column's index", "num", f"v = df['a'].index.values; {_WRITABLE}; v[0] = 99"),
    (
        "another thread",
        "num",
        "import threading\n"
        "    def write():\n"
        f"        v = df['a'].values; {_WRITABLE}; v[0] = 9.0\n"
        "    thread = threading.Thread(target=write)\n"
        "    thread.start()\n"
        "    thread.join()",
    ),
    ("eval in place", "num", "df.eval('a = a + 1', inplace=True)"),
    ("update", "num", "df.update(pd.DataFrame({'a': [9.0]}, index=df.index[:1]))"),
    ("rename in place", "num", "df.rename(columns={'a': 'z'}, inplace=True)"),
    ("sort in place", "num", "df.sort_values('a', ascending=False, inplace=True)"),
    ("drop in place", "num", "df.drop(columns=['b'], inplace=True)"),
    ("a masked assignment", "num", "df[df > 2.0] = 0.0"),
    ("a column slice through loc", "num", "df.loc[:, 'a'] = 0.0"),
    ("isetitem", "num", "df.isetitem(0, [9.0, 9.0, 9.0, 9.0])"),
    ("the index replaced", "num", "df.index = [7, 8, 9, 10]"),
    ("the index names", "num", "df.index.set_names('k', inplace=True)"),
    ("the columns' name", "num", "df.columns.name = 'k'"),
    ("a categorical's categories", "cat", "df['c'] = df['c'].cat.rename_categories({'x': 'q'})"),
    ("a categorical through its accessor", "cat", f"v = df['c'].cat.codes.values; {_WRITABLE}; v[0] = 1"),
    ("a date index's values", "dates", f"v = df.index.values; {_WRITABLE}; v[0] += np.timedelta64(1, 'D')"),
    ("a date index's array", "dates", "df.index.array[0] = pd.Timestamp('1999-01-01')"),
    (
        "a groupby's raw callback",
        "num",
        f"def f(g):\n        v = g['b'].values; {_WRITABLE}; v[0] = 9.0\n        return 0.0\n"
        "    df.groupby('a').apply(f)",
    ),
    (
        "a raw expanding apply",
        "num",
        f"def f(v):\n        {_WRITABLE}\n        v[0] = 9.0\n        return 0.0\n    df['a'].expanding().apply(f, raw=True)",
    ),
    ("pipe to to_numpy", "num", f"v = df.pipe(pd.DataFrame.to_numpy); {_WRITABLE}; v[0, 0] = 9.0"),
)

#: Run in a fresh interpreter; prints one JSON object.
PROBE = """
import json, sys, warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import cash
from cash import _pandas_compat
from cash.decorator import arg_hashing

routes = tuple(map(tuple, json.loads(sys.argv[1])))
real_verdict = None
if sys.argv[2] == "lift":
    _pandas_compat.NEWEST_TESTED = (10**6, 0)

def summary(df):
    return "|".join((
        df.to_json(orient="split", date_format="iso"),
        repr(list(df.index.names)), repr(list(df.columns.names)),
        repr(sorted(df.attrs.items())), repr(df.flags.allows_duplicate_labels),
    ))

@cash.cache
def described(df):
    return summary(df)

def frames():
    made = _pandas_compat.probe_frames(pd, np)
    made["numcols"] = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=pd.Index([7, 8]))
    return made

out = {"pandas": pd.__version__, "routes": {}}
first = frames()["num"]
described(first)
out["memo"] = list(_pandas_compat.frame_memo_state())
out["problems"] = _pandas_compat.self_test(
    pd, np, arg_hashing._probe_remember, arg_hashing._probe_vouched) if out["memo"][0] else None
compiled = _pandas_compat.compile_routes(routes)
for (name, kind, _statements), route in zip(routes, compiled.values()):
    df = frames()[kind]
    before = described(df)
    assert described(df) == before
    remembered = arg_hashing.FRAME_DIGESTS.lookup(df) is not None
    try:
        route(pd, np, df)
    except Exception as exc:
        out["routes"][name] = "refused: " + type(exc).__name__
        if described(df) != summary(df):
            out["routes"][name] = "stale"
        continue
    now = summary(df)
    if described(df) != now:
        out["routes"][name] = "stale"
    elif now == before:
        out["routes"][name] = "no write"
    else:
        out["routes"][name] = "seen" if remembered else "seen, never remembered"
print(json.dumps(out))
"""


def _probe(tmp_path, gate: str) -> dict:
    routes = (*_pandas_compat.WRITE_ROUTES, *MORE_ROUTES)
    done = run_python("-c", PROBE, json.dumps(routes), gate, cwd=tmp_path, timeout=300)
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def probed(tmp_path_factory) -> dict:
    return _probe(tmp_path_factory.mktemp("probe"), "lift")


def test_no_route_serves_the_frame_as_it_was(probed):
    stale = sorted(name for name, verdict in probed["routes"].items() if verdict == "stale")
    assert not stale, f"pandas {probed['pandas']}: a write through {stale} was not seen by the frame memo"


def test_the_start_up_check_passes_on_this_pandas(probed):
    assert probed["memo"] == [True, ""]
    assert probed["problems"] == [], f"pandas {probed['pandas']}"


def test_the_memo_was_in_use_while_probed(probed):
    """A probe of a memo that never remembers a frame proves nothing."""
    verdicts = probed["routes"]
    assert len(verdicts) == len(_pandas_compat.WRITE_ROUTES) + len(MORE_ROUTES)
    seen = [name for name, verdict in verdicts.items() if verdict == "seen"]
    assert len(seen) >= len(verdicts) // 2, verdicts


def test_most_routes_really_write_under_the_tested_pandas(probed):
    if _pandas_compat.pandas_version() > (3, 1):
        pytest.skip("which routes a newer pandas closes is for the probe to find out")
    idle = sorted(name for name, verdict in probed["routes"].items() if not verdict.startswith("seen"))
    assert len(idle) <= 8, probed["routes"]


def test_with_the_gate_down_an_untested_pandas_is_probed_without_the_memo(tmp_path):
    """The same probe as a user's process would run it: on a tested pandas
    the memo is on, on a newer one it is off and every route is seen."""
    verdicts = _probe(tmp_path, "keep")
    tested = _pandas_compat.pandas_version() <= _pandas_compat.NEWEST_TESTED
    assert verdicts["memo"][0] is tested
    assert "stale" not in verdicts["routes"].values()

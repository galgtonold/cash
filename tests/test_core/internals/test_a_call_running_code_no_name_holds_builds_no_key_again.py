"""A call refused because it picks code by a run-time name and ran code no
later process can find again (a closure no name of its module holds) keeps
a refusal verdict, as a call refused for changing its argument does: later
calls, in this process and the next, run uncached without building their
key, but for one probe every `REPROBE_EVERY` uses.

The shape is NiMARE's under a harness that swaps ``cash.cache`` in for
nilearn's cache: a fresh ``cash.cache`` around a bound ``_fit`` on every
fit, which changes ``self`` in a way a hit replays, picks code by a
run-time name, and runs a closure only a library holds (nilearn's module
global ``cache`` there, the record factory ``logging`` keeps here). The
refusal is decided after the full key, and that key moves with a global the body reads (NiMARE's
``__version__``, moved by every commit), so the note that the last call was
not stored never met the next one: each call built its full key, over a
second for a real estimator, to be refused again. The call behaves as plain
Python every time.
"""

from __future__ import annotations

import json
import textwrap

from cash.decorator.refusal_verdicts import REPROBE_EVERY
from tests._scripts import run_python

RUNNERS = """
def run(fn, data):
    return fn(data)
"""

LIB = """
import logging
import os

import runners

VERSION = os.environ["LIB_VERSION"]


def _install():
    made = logging.getLogRecordFactory()

    def record(*args, **kwargs):     # held by logging alone, by no name of lib
        return made(*args, **kwargs)

    logging.setLogRecordFactory(record)


_install()


def _count(data):
    return len(data)


class Est:
    def _fit(self, data):
        self.n_ = len(data)
        name = os.environ.get("RUNNER", "run")
        got = getattr(runners, name)(_count, data)
        logging.getLogger("lib").makeRecord("lib", logging.INFO, __file__, 1, "fit", None, None)
        return got, f"fitted with lib {VERSION}"
"""

JOB = """
import json, sys, warnings
import cash
import lib

c = cash.Cash(cache_dir=sys.argv[1], register_magic=False)

builds = []
real = c._keys.build
c._keys.build = lambda *a, **k: builds.append(1) or real(*a, **k)

results, fitted = [], []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i in range(int(sys.argv[2])):
        est = lib.Est()
        results.append(list(c.cache(est._fit)([0, 0, 0])))
        fitted.append(est.n_)
print(json.dumps({"results": results, "fitted": fitted, "builds": len(builds)}))
"""


def _run(tmp_path, version, calls=3):
    out = run_python("job.py", str(tmp_path / "cache"), str(calls), cwd=tmp_path, env={"LIB_VERSION": version})
    return json.loads(out.stdout.strip().splitlines()[-1])


def _setup(tmp_path):
    (tmp_path / "runners.py").write_text(textwrap.dedent(RUNNERS), encoding="utf-8")
    (tmp_path / "lib.py").write_text(textwrap.dedent(LIB), encoding="utf-8")
    (tmp_path / "job.py").write_text(textwrap.dedent(JOB), encoding="utf-8")


def _plain(version, calls=3):
    return {
        "results": [[3, f"fitted with lib {version}"]] * calls,
        "fitted": [3] * calls,
    }


def test_only_the_first_call_and_the_probes_build_a_key(tmp_path):
    _setup(tmp_path)
    builds = []
    for n in range(4):
        version = f"0.{n}"
        got = _run(tmp_path, version)
        assert {k: got[k] for k in ("results", "fitted")} == _plain(version), got
        builds.append(got["builds"])
    # The first call is refused and keeps the verdict; the eleven after it,
    # across processes and with the key moved in each, build none (before,
    # every one of the twelve built its key).
    assert builds == [1, 0, 0, 0], builds


def test_the_probe_builds_its_key_and_keeps_the_verdict(tmp_path):
    _setup(tmp_path)
    calls = REPROBE_EVERY + 1
    got = _run(tmp_path, "1.0", calls)
    assert {k: got[k] for k in ("results", "fitted")} == _plain("1.0", calls), got
    # The first call, then the probe on the REPROBE_EVERY-th use.
    assert got["builds"] == 2, got
    assert _run(tmp_path, "1.1")["builds"] == 0

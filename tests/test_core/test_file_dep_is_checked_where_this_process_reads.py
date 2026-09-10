"""A file dependency is checked at the path THIS process would read.

CAS-108, three round-17 testers, one mechanism: the dependency was checked at
the path the process that WROTE the entry had resolved.

* **A switched junction.** The tracker recorded ``os.path.realpath`` of the
  path, which resolves the link at write time. Re-point ``current`` at another
  release and every later check stat'ed the OLD target -- still there, still
  unchanged -- so the old answer came back. r17s5 got eleven wrong nightly
  reports this way, including a rollback that returned the NEWER release's
  report; r17s1 got yesterday's data through a re-pointed ``current``.
* **Two installs of one package.** Byte-identical code shares cache keys, and
  the entry recorded its package data at the writer's install path. Another
  install validated the writer's ``rates.json`` and was served the writer's
  figures; an install with no ``rates.json`` at all was served a value
  instead of raising (r17s2).

Every step runs in a FRESH process against one shared cache, because that is
the shape the testers hit and the one a cache exists for.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.core

_READER = textwrap.dedent("""
    import json, sys, time
    import cash

    @cash.cache
    def total(path):
        print("RAN", file=sys.stderr, flush=True)
        time.sleep(0.25)                      # over the persistence floor
        with open(path, encoding="utf-8") as fh:
            return sum(int(line) for line in fh if line.strip())

    print(json.dumps(total(sys.argv[1])))
""")


def _run(script, *argv, cache, cwd, extra_path=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(cache)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra_path:
        env["PYTHONPATH"] = str(extra_path)
    out = subprocess.run([sys.executable, str(script), *map(str, argv)],
                         cwd=str(cwd), capture_output=True, text=True, env=env)
    return out


def _point(link, target):
    """Create or re-point *link* at *target*: a junction on Windows."""
    if os.path.lexists(link):
        if os.name == "nt":
            os.rmdir(link)                    # removes the junction, not the target
        else:
            os.unlink(link)
    if os.name == "nt":
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                              capture_output=True, text=True)
        if made.returncode != 0:
            pytest.skip(f"cannot create a junction here: {made.stdout}{made.stderr}")
    else:
        os.symlink(target, link, target_is_directory=True)


def _release(root, name, values, mtime_ns=1_000_000_000_000_000_000):
    """A release directory whose ref file has fixed size AND fixed mtime.

    Same size and same mtime across releases on purpose: that is what a
    reproducible deploy lays down, and it is the arm that defeats every
    timestamp-based check.
    """
    d = root / name
    d.mkdir()
    ref = d / "ref.txt"
    ref.write_text("\n".join(str(v) for v in values) + "\n", encoding="utf-8")
    os.utime(ref, ns=(mtime_ns, mtime_ns))
    return d


def test_a_switched_junction_is_seen(tmp_path):
    """THE BUG: switch `current`, get the previous target's answer."""
    (tmp_path / "reader.py").write_text(_READER, encoding="utf-8")
    rel_a = _release(tmp_path, "relA", [10, 20, 30])        # 60
    rel_b = _release(tmp_path, "relB", [40, 20, 30])        # 90
    # different content, SAME size, SAME mtime -- nothing timestamp-based can tell
    assert (rel_a / "ref.txt").stat().st_size == (rel_b / "ref.txt").stat().st_size
    current = tmp_path / "current"
    cache = tmp_path / "cache"
    target = current / "ref.txt"

    _point(current, rel_a)
    first = _run(tmp_path / "reader.py", target, cache=cache, cwd=tmp_path)
    assert json.loads(first.stdout) == 60, first.stderr

    _point(current, rel_b)
    second = _run(tmp_path / "reader.py", target, cache=cache, cwd=tmp_path)

    assert json.loads(second.stdout) == 90, (
        "a re-pointed junction served the previous target's answer"
    )
    assert "RAN" in second.stderr


def test_a_rollback_gets_the_older_answer(tmp_path):
    """A -> B -> A must return A's answer the third time, not B's."""
    (tmp_path / "reader.py").write_text(_READER, encoding="utf-8")
    rel_a = _release(tmp_path, "relA", [1, 2, 3])            # 6
    rel_b = _release(tmp_path, "relB", [7, 8, 9])            # 24
    current = tmp_path / "current"
    cache = tmp_path / "cache"
    target = current / "ref.txt"

    answers = []
    for release in (rel_a, rel_b, rel_a):
        _point(current, release)
        out = _run(tmp_path / "reader.py", target, cache=cache, cwd=tmp_path)
        assert out.returncode == 0, out.stderr
        answers.append(json.loads(out.stdout))

    assert answers == [6, 24, 6], answers


def test_an_unswitched_junction_still_hits(tmp_path):
    """The control: the twin dependency must not cost a real hit."""
    (tmp_path / "reader.py").write_text(_READER, encoding="utf-8")
    rel_a = _release(tmp_path, "relA", [1, 2, 3])
    current = tmp_path / "current"
    cache = tmp_path / "cache"
    _point(current, rel_a)

    _run(tmp_path / "reader.py", current / "ref.txt", cache=cache, cwd=tmp_path)
    again = _run(tmp_path / "reader.py", current / "ref.txt", cache=cache, cwd=tmp_path)

    assert json.loads(again.stdout) == 6
    assert "RAN" not in again.stderr, "an untouched junction target recomputed"


# --------------------------------------------------------------------------- #
# Two installs of one package, one cache                                      #
# --------------------------------------------------------------------------- #

_PKG_INIT = textwrap.dedent("""
    import json, sys, time
    from importlib import resources
    import cash

    def _rate(cur):
        return json.loads(resources.files("fxpkg").joinpath("rates.json").read_text())[cur]

    @cash.cache
    def eur_total(n):
        print("RAN", file=sys.stderr, flush=True)
        time.sleep(0.25)
        return round(n * _rate("EUR"), 4)
""")

_RUN_PKG = "import fxpkg, json, sys\nprint(json.dumps(fxpkg.eur_total(1000)))\n"


def _site(root, name, rates):
    site = root / name
    pkg = site / "fxpkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(_PKG_INIT, encoding="utf-8")
    if rates is not None:
        (pkg / "rates.json").write_text(json.dumps(rates), encoding="utf-8")
    (site / "run.py").write_text(_RUN_PKG, encoding="utf-8")
    return site


def test_another_install_checks_its_own_package_data(tmp_path):
    """THE BUG: install B was served install A's exchange rate."""
    cache = tmp_path / "cache"
    site_a = _site(tmp_path, "siteA", {"EUR": 0.85})
    site_b = _site(tmp_path, "siteB", {"EUR": 0.92})

    a = _run(site_a / "run.py", cache=cache, cwd=site_a, extra_path=site_a)
    b = _run(site_b / "run.py", cache=cache, cwd=site_b, extra_path=site_b)

    assert json.loads(a.stdout) == 850.0, a.stderr
    assert json.loads(b.stdout) == 920.0, "install B was served install A's figures"
    assert "RAN" in b.stderr


def test_an_install_missing_its_data_raises(tmp_path):
    """Not a value computed from a file this install does not have."""
    cache = tmp_path / "cache"
    site_a = _site(tmp_path, "siteA", {"EUR": 0.85})
    site_c = _site(tmp_path, "siteC", None)

    _run(site_a / "run.py", cache=cache, cwd=site_a, extra_path=site_a)
    c = _run(site_c / "run.py", cache=cache, cwd=site_c, extra_path=site_c)

    assert c.returncode != 0, f"served {c.stdout.strip()!r} without a rates.json"
    assert "FileNotFoundError" in c.stderr


def test_identical_installs_still_share(tmp_path):
    """The control: same package data in both installs is a hit, as before."""
    cache = tmp_path / "cache"
    site_a = _site(tmp_path, "siteA", {"EUR": 0.85})
    site_b = _site(tmp_path, "siteB", {"EUR": 0.85})

    _run(site_a / "run.py", cache=cache, cwd=site_a, extra_path=site_a)
    b = _run(site_b / "run.py", cache=cache, cwd=site_b, extra_path=site_b)

    assert json.loads(b.stdout) == 850.0
    assert "RAN" not in b.stderr, "identical installs stopped sharing"

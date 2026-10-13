"""A file beside a script is looked for beside THAT script, nowhere else.

A file under the script's own directory is recorded relative to the code, so
another copy of the same script (a second release beside the first) checks its
own copy. It was recorded against ``__main__``, and ``__main__`` names a
different script in every process: ``cash inspect``, ``python -m pytest``, or a
script that imports the job. Those relocated the file beside their own
``__main__``. A miss reason named ``…/site-packages/pytest/data/prices.csv
(file missing)``, and a script importing the job checked its OWN ``data/``
folder: a stale copy there served the old answer after the job's file changed.
"""

from __future__ import annotations

import os
import sys
import textwrap
import types

import pytest

from cash._paths import normalize_path
from cash.decorator.cache_metadata import CacheMetadata
from cash.decorator.explain import describe_stale_files
from cash.tracking.file_dep_snapshot import dep_is_fresh, dep_path_for_this_process
from tests._scripts import CASH_SRC, run_python

pytestmark = pytest.mark.core

_JOB = textwrap.dedent("""
    import pathlib, sys
    import cash

    @cash.cache(assume_safe=True)
    def load():
        print("RAN", file=sys.stderr, flush=True)
        return (pathlib.Path(__file__).parent / "data" / "prices.csv").read_text(encoding="utf-8")

    if __name__ == "__main__":
        print(load())
""")

# Imports the job and calls it: the same cache key as `python job.py`.
_OTHER = textwrap.dedent("""
    import job
    print(job.load())
""")

_FIXED_NS = 1_000_000_000_000_000_000


def _write(path, text):
    """Same size for every version, and one mtime per text: two copies of a
    version look alike, two versions do not (Windows keeps no change time)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    stamp = _FIXED_NS + sum(text.encode()) * 1_000_000_000
    os.utime(path, ns=(stamp, stamp))


def _job(tmp_path):
    job_dir = tmp_path / "job"
    _write(job_dir / "data" / "prices.csv", "v1\n")
    (job_dir / "job.py").write_text(_JOB, encoding="utf-8")
    first = run_python("job.py", cwd=job_dir, cache_dir=tmp_path / "cache")
    assert first.stdout.strip() == "v1", first.stderr
    return job_dir


def _recorded(tmp_path):
    from cash.__main__ import _store

    (entry,) = [s.metadata for s in _store(tmp_path / "cache").entries()]
    return entry["auto_file_deps"]


def test_a_script_importing_the_job_checks_the_jobs_file(tmp_path):
    """THE BUG: a stale `data/prices.csv` beside the importer served `v1`."""
    job_dir = _job(tmp_path)
    other_dir = tmp_path / "other"
    _write(other_dir / "data" / "prices.csv", "v1\n")  # an old copy beside the importer
    (other_dir / "other.py").write_text(_OTHER, encoding="utf-8")
    _write(job_dir / "data" / "prices.csv", "v2\n")

    env = {"PYTHONPATH": os.pathsep.join([CASH_SRC, str(job_dir)])}
    second = run_python("other.py", cwd=other_dir, cache_dir=tmp_path / "cache", env=env)

    assert second.stdout.strip() == "v2", "the importer's own data/ folder decided the job's entry"
    assert "RAN" in second.stderr


def test_another_copy_of_the_script_checks_its_own_file(tmp_path):
    """The control: a second release of the job still reads its own data."""
    _job(tmp_path)
    release = tmp_path / "release2"
    _write(release / "data" / "prices.csv", "v9\n")
    (release / "job.py").write_text(_JOB, encoding="utf-8")

    second = run_python("job.py", cwd=release, cache_dir=tmp_path / "cache")

    assert second.stdout.strip() == "v9", second.stderr
    assert "RAN" in second.stderr


def test_the_same_script_still_hits(tmp_path):
    job_dir = _job(tmp_path)
    again = run_python("job.py", cwd=job_dir, cache_dir=tmp_path / "cache")
    assert again.stdout.strip() == "v1"
    assert "RAN" not in again.stderr


@pytest.fixture
def other_main(tmp_path, monkeypatch):
    """This process runs another script, as `python -m pytest` or `cash inspect` does."""
    script = tmp_path / "elsewhere" / "run.py"
    main = types.ModuleType("__main__")
    main.__file__ = str(script)
    main.__spec__ = None
    monkeypatch.setitem(sys.modules, "__main__", main)
    monkeypatch.delitem(sys.modules, "__mp_main__", raising=False)
    monkeypatch.delitem(sys.modules, "job", raising=False)
    return script.parent


def test_a_miss_names_the_recorded_path_under_another_main(tmp_path, other_main):
    job_dir = _job(tmp_path)
    recorded = _recorded(tmp_path)
    prices = normalize_path(str(job_dir.resolve() / "data" / "prices.csv"))
    _write(other_main / "data" / "prices.csv", "v1\n")
    _write(job_dir / "data" / "prices.csv", "v2\n")

    assert dep_path_for_this_process(prices, recorded[prices]) == prices
    resolved, fresh, _ = dep_is_fresh(prices, recorded[prices])
    assert (resolved, fresh) == (prices, False)
    said = describe_stale_files(CacheMetadata(auto_file_deps=recorded))
    assert said.startswith(prices), said


def test_an_entry_naming_only_main_is_checked_where_it_was_written(tmp_path, other_main):
    """Entries written before the script was named say `__main__`: no relocation."""
    job_dir = _job(tmp_path)
    recorded = _recorded(tmp_path)
    prices = normalize_path(str(job_dir.resolve() / "data" / "prices.csv"))
    legacy = dict(recorded[prices], code_mod="__main__")

    assert dep_path_for_this_process(prices, legacy) == prices

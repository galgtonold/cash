"""cash's own configuration reads never become a cached function's dependency.

Found while fixing CASH_DISABLE. A cached function that
calls another cached function recorded, as its OWN file dependencies, the
project's pyproject.toml, the user config file (as absent) and every
`.git` / `setup.py` / `setup.cfg` looked for on the way up the tree. The nested
call's bookkeeping resolved cash's configuration while the outer call's file
tracker was live, and the tracker cannot tell cash's reads from the user's --
the storage-path filter only knows cache files. So bumping the project's
version, creating a user config or running `git init` recomputed every such
function. Extra recomputes, never a stale value.

And in the same spot: the threshold that decides which files are hashed in
full came from the process-wide singleton, or a fresh read of the config
files, never from the `Cash` instance doing the work -- so
`Cash(file_hash_full_max_bytes=...)` on your own instance was ignored.

The tracker's own filters are cash's reads too. The first read it classifies
in a process looks up which time zone database directories exist, and it
asked through the same `os.path.isdir` it watches, so every missing
`zoneinfo.TZPATH` entry (`/usr/lib/zoneinfo`, `/etc/zoneinfo` ...) became an
absent-file dependency of the entry being computed: whichever cached call
made the first read in a process, never a later one.
"""

from __future__ import annotations

import os
import time

import pytest

from cash import Cash, FileBackend
from cash.tracking import read_classification

pytestmark = pytest.mark.core


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project directory with a pyproject.toml, as the cwd."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname="p"\nversion="1"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _nested(c, data):
    @c.cache
    def load():
        with open(data, encoding="utf-8") as f:
            return f.read().strip()

    @c.cache(depends_on=[load])
    def upper():
        return load().upper()

    return load, upper


def test_the_outer_entry_depends_on_the_data_file_only(project):
    data = project / "data.txt"
    data.write_text("hello", encoding="utf-8")
    c = Cash(backend=FileBackend(cache_dir=str(project / "cache")), register_magic=False)
    _, upper = _nested(c, str(data))
    upper()
    meta, _ = c.backend.get(upper.explain().cache_key)
    deps = [os.path.basename(p) for p in (meta or {}).get("auto_file_deps", {})]
    assert deps == ["data.txt"], deps


@pytest.fixture
def first_read_in_process():
    """Make the next classified read the first in this process, as it is in a
    fresh kernel or worker: the time zone directories are looked up again."""
    read_classification._tz_roots.cache_clear()


def _deps(c, fn):
    meta, _ = c.backend.get(fn.explain().cache_key)
    return sorted(os.path.basename(p) for p in (meta or {}).get("auto_file_deps", {}))


def test_the_first_read_in_a_process_adds_no_time_zone_directories(project, first_read_in_process):
    data = project / "data.txt"
    data.write_text("hello", encoding="utf-8")
    c = Cash(backend=FileBackend(cache_dir=str(project / "cache")), register_magic=False)
    _, upper = _nested(c, str(data))
    upper()
    assert _deps(c, upper) == ["data.txt"]


def test_a_missing_directory_the_user_checks_still_counts(project, first_read_in_process):
    """The control: the inner call's own `isdir` on a missing path is still
    recorded on the outer entry."""
    data = project / "data.txt"
    data.write_text("hello", encoding="utf-8")
    c = Cash(backend=FileBackend(cache_dir=str(project / "cache")), register_magic=False)

    @c.cache
    def load():
        with open(data, encoding="utf-8") as f:
            text = f.read().strip()
        return text if os.path.isdir(project / "extra") else text + "!"

    @c.cache(depends_on=[load])
    def upper():
        return load().upper()

    assert upper() == "HELLO!"
    assert _deps(c, upper) == ["data.txt", "extra"]


def test_editing_pyproject_does_not_recompute_it(project):
    data = project / "data.txt"
    data.write_text("hello", encoding="utf-8")
    c = Cash(backend=FileBackend(cache_dir=str(project / "cache")), register_magic=False)
    _, upper = _nested(c, str(data))
    upper()
    (project / "pyproject.toml").write_text('[project]\nname="p"\nversion="2"\n', encoding="utf-8")
    assert upper.explain().reason == "hit"


def test_a_real_nested_read_still_propagates(project):
    """The control: the user's own file, read by the inner call, still counts."""
    data = project / "data.txt"
    data.write_text("hello", encoding="utf-8")
    c = Cash(backend=FileBackend(cache_dir=str(project / "cache")), register_magic=False)
    _, upper = _nested(c, str(data))
    assert upper() == "HELLO"
    time.sleep(0.02)
    data.write_text("world", encoding="utf-8")
    assert upper() == "WORLD"


def test_an_instance_threshold_is_used(tmp_path):
    """`file_hash_full_max_bytes` on your own Cash: a file above it is sampled,
    so a touch (its timestamp backstop) recomputes -- it did not."""
    big = tmp_path / "big.bin"
    big.write_bytes(os.urandom(300_000))
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False, file_hash_full_max_bytes=1000)
    runs = []

    @c.cache(assume_safe=True)
    def size(path):
        runs.append(1)
        with open(path, "rb") as f:
            return len(f.read())

    size(str(big))
    time.sleep(0.02)
    os.utime(big)
    size(str(big))
    assert len(runs) == 2, "the instance's threshold was ignored"

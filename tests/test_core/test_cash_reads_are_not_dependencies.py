"""cash's own configuration reads never become a cached function's dependency.

Found while fixing CASH_DISABLE (round-17 backlog). A cached function that
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
"""
from __future__ import annotations

import os
import time

import pytest

from cash import Cash, FileBackend

pytestmark = pytest.mark.core


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project directory with a pyproject.toml, as the cwd."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname="p"\nversion="1"\n',
                                             encoding="utf-8")
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


def test_editing_pyproject_does_not_recompute_it(project):
    data = project / "data.txt"
    data.write_text("hello", encoding="utf-8")
    c = Cash(backend=FileBackend(cache_dir=str(project / "cache")), register_magic=False)
    _, upper = _nested(c, str(data))
    upper()
    (project / "pyproject.toml").write_text('[project]\nname="p"\nversion="2"\n',
                                            encoding="utf-8")
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
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False,
             file_hash_full_max_bytes=1000)
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

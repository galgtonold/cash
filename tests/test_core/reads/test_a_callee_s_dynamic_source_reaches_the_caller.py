"""What a cached callee's ``dynamic_depends_on=`` resolves to reaches its cached callers.

The sources are folded into the callee's own key, which a caller never sees:
``report()`` calling ``load("data.txt")`` kept serving its old result after
``data.txt`` changed, while ``load`` called directly recomputed. A file
source becomes a file the caller's entry checks; any other source is one
only a call of the callee can check, so the caller's result is not stored.
"""

from __future__ import annotations

import textwrap
import warnings

from cash import Cash, DataSource, InMemoryBackend
from tests._files import rewrite
from tests._scripts import run_python

JOB = """
    import subprocess, sys
    import cash
    from cash import FileDataSource

    # Read in a child process, which cash does not see: the declaration is
    # all that ties the result to the file.
    @cash.cache(dynamic_depends_on=lambda name: FileDataSource(name), assume_safe=True)
    def load(name):
        code = f"print(open({name!r}).read().strip())"
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()

    @cash.cache
    def report():
        print("[REPORT]", file=sys.stderr)  # @cash:assume-safe
        return "report:" + load("data.txt")

    print(report())
"""


def test_a_file_source_of_the_callee_is_checked_by_the_caller(tmp_path):
    (tmp_path / "job.py").write_text(textwrap.dedent(JOB), encoding="utf-8")
    data = tmp_path / "data.txt"
    data.write_text("v1", encoding="utf-8")
    first = run_python("job.py", cwd=tmp_path)
    again = run_python("job.py", cwd=tmp_path)
    assert first.stdout.strip() == again.stdout.strip() == "report:v1"
    assert "[REPORT]" not in again.stderr  # unchanged: the caller's entry is served
    rewrite(data, "v2-longer")
    assert run_python("job.py", cwd=tmp_path).stdout.strip() == "report:v2-longer"


class Version(DataSource):
    def __init__(self):
        self.version = 1

    def get_id(self) -> str:
        return "version"

    def state_token(self):
        return self.version


def test_another_source_leaves_the_caller_unstored_with_a_warning():
    c = Cash(backend=InMemoryBackend())
    source = Version()
    runs = []

    @c.cache(dynamic_depends_on=lambda: source, assume_safe=True)
    def load():
        return source.version

    @c.cache(assume_safe=True)
    def report():
        runs.append(1)
        return load() * 10

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert report() == 10
    assert any("STORE-UNTRACKED-SOURCE" in str(w.message) for w in rec)
    source.version = 2
    assert report() == 20
    assert len(runs) == 2

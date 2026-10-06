"""What a cached callee's ``dynamic_depends_on=`` resolves to reaches its cached callers.

The sources are folded into the callee's own key, which a caller never sees:
``report()`` calling ``load("data.txt")`` kept serving its old result after
``data.txt`` changed, while ``load`` called directly recomputed. A file
source becomes a file the caller's entry checks. Any other source is kept in
the caller's entry with its token, pickled so a later process can ask it, and
every lookup of the caller asks it again. One that cannot be pickled keeps
the entry in this process's RAM, or unstored, with STORE-UNTRACKED-SOURCE.
"""

from __future__ import annotations

import textwrap
import threading
import warnings

from cash import DataSource
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


class Counter(DataSource):
    """A source whose token is its own state. Reached only through the
    resolver, so nothing else ties the caller's key to it."""

    def __init__(self):
        self.n = 1

    def get_id(self) -> str:
        return "counter"

    def state_token(self):
        return self.n


SOURCES: dict = {}


def _caller(c, source, runs):
    SOURCES["a"] = source

    @c.cache(dynamic_depends_on=lambda name: SOURCES[name], assume_safe=True)
    def load(name):
        return name.upper()

    @c.cache(assume_safe=True)
    def report():
        runs.append(1)
        return load("a")

    return report


def test_the_caller_hits_while_the_source_is_unchanged_and_recomputes_after(cash_instance):
    runs: list = []
    source = Counter()
    report = _caller(cash_instance, source, runs)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        report()
        report()
        assert len(runs) == 1
        source.n = 2
        report()
        assert len(runs) == 2
        report()
        assert len(runs) == 2
    assert not any("STORE-UNTRACKED-SOURCE" in str(w.message) for w in rec)
    assert report.cache_info()["miss_reasons"].get("dynamic dependency changed") == 1


class Raising(Counter):
    def state_token(self):
        if self.n > 1:
            raise RuntimeError("server down")
        return self.n


def test_a_token_that_raises_at_lookup_recomputes(cash_instance):
    runs: list = []
    source = Raising()
    report = _caller(cash_instance, source, runs)
    report()
    source.n = 2
    # The callee's own key cannot be built either: it runs uncached.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report()
    assert len(runs) == 2


TABLE_JOB = """
    import os, subprocess, sys
    import cash
    from cash import DataSource

    class Table(DataSource):
        # Its version lives outside the process (here, the environment).
        def __init__(self, name):
            self.name = name

        def get_id(self):
            return "table:" + self.name

        def state_token(self):
            return os.environ["TABLE_VERSION"]

    # Reads the table in a way cash cannot see: the declaration is all that
    # ties the result to it.
    @cash.cache(dynamic_depends_on=lambda name: Table(name), assume_safe=True)
    def load(name):
        code = "import os; print(os.environ['TABLE_VERSION'])"
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()

    @cash.cache(assume_safe=True)
    def report():
        print("[REPORT]", file=sys.stderr)
        return "report:" + load("prices")

    print(report())
"""


def test_the_source_is_kept_with_the_entry_across_a_restart(tmp_path):
    (tmp_path / "job.py").write_text(textwrap.dedent(TABLE_JOB), encoding="utf-8")
    first = run_python("job.py", cwd=tmp_path, env={"TABLE_VERSION": "1"})
    again = run_python("job.py", cwd=tmp_path, env={"TABLE_VERSION": "1"})
    assert first.stdout.strip() == again.stdout.strip() == "report:1"
    assert "[REPORT]" in first.stderr
    assert "[REPORT]" not in again.stderr, again.stderr  # unchanged: served from disk
    assert "STORE-UNTRACKED-SOURCE" not in first.stderr
    moved = run_python("job.py", cwd=tmp_path, env={"TABLE_VERSION": "2"})
    assert moved.stdout.strip() == "report:2"
    assert "[REPORT]" in moved.stderr


class Unpicklable(Counter):
    """Holds a lock, as a source holding a connection would: no pickle."""

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()


def test_an_unpicklable_source_keeps_the_entry_in_ram_with_a_warning(disk_cash):
    runs: list = []
    source = Unpicklable()
    report = _caller(disk_cash, source, runs)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        report()
        report()
    found = [str(w.message) for w in rec if "STORE-UNTRACKED-SOURCE" in str(w.message)]
    assert found and "kept in memory for this process only" in found[0], found
    assert len(runs) == 1  # served from RAM in this process
    source.n = 2
    report()
    assert len(runs) == 2
    on_disk = [m.get("func_name") or "" for m in disk_cash.backend.backends[1].list_entries()]
    assert any(name.endswith(".load") for name in on_disk), on_disk  # the callee's own entry
    assert not any(name.endswith(".report") for name in on_disk), on_disk


def test_an_unpicklable_source_is_not_stored_without_a_ram_tier(cash_with_file_backend):
    runs: list = []
    report = _caller(cash_with_file_backend, Unpicklable(), runs)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        report()
        report()
    found = [str(w.message) for w in rec if "STORE-UNTRACKED-SOURCE" in str(w.message)]
    assert found and "returned but not cached" in found[0], found
    assert len(runs) == 2


def test_a_caller_served_from_its_entry_passes_the_source_on(cash_instance):
    """``outer()`` first runs when ``report()`` already has an entry: the hit
    runs no callee, so the entry hands its sources to ``outer`` itself."""
    runs: list = []
    source = Counter()
    report = _caller(cash_instance, source, runs)
    report()
    outer_runs = []

    @cash_instance.cache(assume_safe=True)
    def outer():
        outer_runs.append(1)
        return report()

    outer()
    outer()
    assert len(outer_runs) == 1
    source.n = 2
    outer()
    assert len(outer_runs) == 2

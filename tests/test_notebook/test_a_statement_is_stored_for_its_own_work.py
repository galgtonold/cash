"""A statement around a cached call is stored for its own work, not the call's.

``b = shifted(a) + 1`` over a 0.2 s call stored the call's result twice: as the
call's entry, and inside the statement's value, because the statement was
priced at the call's cost too. The call's entry already saves that. What
storing the statement's value saves is the ``+ 1`` and restoring the call's
result, so it is stored only when that is worth it; otherwise the call is
served from the cache and the cheap rest runs again. The statement's recorded
cost, which a hit is credited with, stays the whole cost.
"""

from __future__ import annotations

import pytest

from cash.backends import FileBackend, InMemoryBackend, TieredBackend
from tests._cell_driver import run_cash_cell

DEFS = (
    "import time\n"
    "def shifted(x):\n"
    "    time.sleep(0.2)\n"
    "    return list(x)\n"
    "def load(p):\n"
    "    time.sleep(0.2)\n"
    "    with open(p) as fh:\n"
    "        return fh.read()\n"
)


@pytest.fixture
def tiers(cash_magics, cash_instance, tmp_path):
    """``cash_magics`` over RAM and a disk tier, as ``%cash_on`` builds it: a
    value that is not stored still leaves a metadata-only record on disk."""
    ram, disk = InMemoryBackend(), FileBackend(str(tmp_path / "nbcache"), flush_interval=0)
    cash_instance.backend = TieredBackend([ram, disk])
    yield cash_magics, ram, disk
    disk.shutdown()


def _entry(backend, needle):
    """``(metadata, value)`` of the statement whose code holds *needle*, or
    ``(None, None)``."""
    for meta in backend.list_entries() or ():
        if needle in str(meta.get("code", "")):
            return backend.get(meta["key"])
    return None, None


def _metadata(disk, needle):
    """The disk tier's record of the statement whose code holds *needle*."""
    disk._writes.wait_all()
    for meta in disk.list_entries() or ():
        if needle in str(meta.get("code", "")):
            return meta
    raise AssertionError(f"no record of {needle!r}")


def _setup(magics):
    run_cash_cell(magics, DEFS)
    run_cash_cell(magics, "a = list(range(1000))")


def test_a_cheap_rest_around_a_cached_call_keeps_no_value(tiers, mock_shell):
    """F: ``b = shifted(a) + [1]``."""
    magics, ram, disk = tiers
    _setup(magics)
    run_cash_cell(magics, "b = shifted(a) + [1]")
    assert mock_shell.user_ns["b"] == [*range(1000), 1]
    assert _entry(ram, "shifted(a) + [1]") == (None, None), "the statement kept a second copy of the call's result"
    meta = _metadata(disk, "shifted(a) + [1]")
    assert meta.get("metadata_only"), meta
    assert meta["execution_time"] >= 0.2, "a hit is credited with the whole cost"
    assert meta["store_time"] < 0.1, meta
    assert meta.get("output_lineages", {}).get("b"), "the lineage a restart needs was not kept"

    # Run again: the call is served, the rest recomputes, the value is right.
    mock_shell.user_ns.pop("b")
    run_cash_cell(magics, "b = shifted(a) + [1]")
    assert mock_shell.user_ns["b"] == [*range(1000), 1]


def test_a_statement_missing_while_its_call_hits_keeps_no_value(tiers, mock_shell):
    """G: a second statement over the same call is served the call."""
    magics, ram, disk = tiers
    _setup(magics)
    run_cash_cell(magics, "b = shifted(a) + [1]")
    run_cash_cell(magics, "c = shifted(a) + [2]")
    assert mock_shell.user_ns["c"] == [*range(1000), 2]
    assert _entry(ram, "shifted(a) + [2]") == (None, None)
    meta = _metadata(disk, "shifted(a) + [2]")
    assert meta["execution_time"] >= 0.2, "the served call's compute was left out of the credit"


def test_a_statement_whose_own_work_is_expensive_is_stored(tiers, mock_shell):
    """C: the statement's own work is worth keeping, so it is stored too."""
    magics, ram, disk = tiers
    _setup(magics)
    code = "b = shifted(a) + [time.sleep(0.2)]"
    run_cash_cell(magics, code)
    meta, value = _entry(ram, code)
    assert value["variables"]["b"][-1] is None
    assert meta["store_time"] >= 0.15, meta


def test_a_plain_call_result_keeps_its_reference_entry(tiers, mock_shell):
    """A: ``b = shifted(a)`` does nothing of its own, but its entry holds only a
    reference to the call's, and lets a later run restore ``b`` without running."""
    magics, ram, disk = tiers
    from cash.notebook.call_refs import CallRef

    _setup(magics)
    run_cash_cell(magics, "b = shifted(a)")
    meta, value = _entry(ram, "b = shifted(a)")
    assert isinstance(value["variables"]["b"], CallRef)


def test_persist_still_stores_the_value(tiers, mock_shell):
    magics, ram, disk = tiers
    _setup(magics)
    code = "# @cash:persist\nb = shifted(a) + [1]"
    run_cash_cell(magics, code)
    meta, value = _entry(ram, "shifted(a) + [1]")
    assert value["variables"]["b"][-1] == 1


def test_a_file_read_inside_a_cached_call_does_not_waive_the_floor(tiers, mock_shell, tmp_path):
    """The floor is waived for a statement that reads a file, because the read
    is the work. A read inside a cached call is the call's work, and its entry
    keeps the file as a dependency."""
    magics, ram, disk = tiers
    path = tmp_path / "data.txt"
    path.write_text("hello", encoding="utf-8")
    _setup(magics)
    run_cash_cell(magics, f"t = load(r'{path}') + '!'")
    assert mock_shell.user_ns["t"] == "hello!"
    assert _entry(ram, "load(") == (None, None)
    assert _metadata(disk, "load(").get("metadata_only")

    # The control: the same read made by the statement itself waives it.
    run_cash_cell(magics, f"u = open(r'{path}').read() + '!'")
    meta, value = _entry(ram, "open(")
    assert value["variables"]["u"] == "hello!"


def test_a_cheap_statement_over_it_is_not_kept_for_rebuilding_the_call(tiers, mock_shell):
    """A cheap statement over a costly input gets an entry, so the end of the
    cell can write it to disk. Rebuilding ``b`` costs the ``+ [1]`` and a
    restore of the call's result, not the call, so ``n = len(b)`` gets none."""
    magics, ram, disk = tiers
    _setup(magics)
    run_cash_cell(magics, "b = shifted(a) + [1]")
    run_cash_cell(magics, "n = len(b)")
    assert mock_shell.user_ns["n"] == 1001
    assert _entry(ram, "len(b)") == (None, None)
    with pytest.raises(AssertionError):
        _metadata(disk, "len(b)")


def test_under_cache_fit_the_fit_is_the_statements_own_work(cash_magics, clean_backend):
    """A fit is a method call, never a call unit: under ``cache-fit`` it is the
    statement's own work, and the statement is stored for it however much of
    its time a cached call took."""
    pytest.importorskip("sklearn")
    import ast

    from cash.analysis.annotations import get_statement_annotations
    from cash.notebook.cache_status import CacheStatus

    magics = cash_magics
    ns = magics.shell.user_ns

    def run(code):
        node = ast.parse(code).body[0]
        return magics._statement_processor.process_statement(
            ast.unparse(node), annotation=get_statement_annotations(code, node)
        )

    run_cash_cell(
        magics,
        "import time\nfrom sklearn.feature_extraction.text import TfidfVectorizer\n"
        "def prep(n):\n    time.sleep(0.2)\n    return [f'the cat {i % 97} sat by dog {i % 89}' for i in range(n)]",
    )
    run("vec = TfidfVectorizer()")
    code = "# @cash:cache-fit\nX = vec.fit_transform(prep(40_000))"
    assert run(code)["status"] == CacheStatus.COMPUTED
    meta, value = _entry(clean_backend, "fit_transform")
    assert value is not None, "the fit was priced as the cached call's time"

    run("vec = TfidfVectorizer()")
    del ns["X"]
    again = run(code)
    assert again["status"] == CacheStatus.RESTORED, {
        k: again.get(k) for k in ("miss_reason", "skipped_reason", "uncacheable_reasons")
    }
    assert hasattr(ns["vec"], "vocabulary_") and ns["X"].shape[0] == 40_000


def test_a_call_under_the_persistence_floor_is_the_statements_work(tiers, mock_shell):
    """A call quicker than the persistence floor is kept in RAM only, so after a
    restart only the statement's own entry brings its result back: it is not
    left out of what storing the statement saves."""
    magics, ram, disk = tiers
    _setup(magics)
    run_cash_cell(magics, "def quick(x):\n    time.sleep(0.05)\n    return list(x)")
    run_cash_cell(magics, "b = quick(a) + [1]")
    meta, value = _entry(ram, "quick(a) + [1]")
    assert value is not None and value["variables"]["b"][-1] == 1
    assert meta["store_time"] >= 0.04, meta


def test_a_call_whose_result_a_method_is_called_on_is_a_call_unit(tiers, mock_shell):
    """``b = shifted(a).count(5)``: the rewrite took the bound method
    ``shifted(a).count`` as the callee, which is never cached, so ``shifted(a)``
    inside it ran again on every run. It is the call unit; the ``.count(5)``
    is the statement's own, cheap, work."""
    magics, ram, disk = tiers
    _setup(magics)
    run_cash_cell(magics, "b = shifted(a).count(5)")
    assert mock_shell.user_ns["b"] == 1
    calls = [m for m in ram.list_entries() or () if str(m.get("key", "")).startswith("call:")]
    assert [str(m.get("function")).rsplit(".", 1)[-1] for m in calls] == ["shifted"], calls
    assert _entry(ram, "count(5)") == (None, None), "the statement kept a value its call already holds"

"""With the trace on, an entry is served only when the trace it carries was
checked and agreed. An entry with no trace to check, or whose check could
not run, runs again and stores one.

A missing trace counted as fine: entries stored before traces were kept
where they are read now had none, and were served on their key alone for
good (a hit writes no trace). A function given new code outside any
statement, the edit the trace exists for, then went unseen. And when the
cache's ``local_dir`` raised, the check never started, and stale hits were
served with nothing logged.

Each entry now carries its own trace in its metadata, so the trace goes
wherever the entry goes, on every backend. Before, traces were kept in a
file per cache folder, and caches with no folder (SQLite, Redis, S3) shared
one under the temp dir, keyed by cache key alone: another notebook's trace
for the same key vouched for this one's stale entry.
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap

import pytest

from cash.analysis.annotations import CacheAnnotation
from cash.backends import FileBackend
from cash.notebook import shadow
from cash.notebook.cache_status import CacheStatus

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="traces are recorded on 3.12+")

_PERSIST = CacheAnnotation(persist=True)

_IMPL = """
import time

def inner(x):
    return x * 2

def slow_inner(x, b=0):
    time.sleep(0.25)
    return inner(x) + b
"""


@pytest.fixture
def trace_on(tmp_path, monkeypatch, request):
    monkeypatch.delenv("CASH_SHADOW", raising=False)
    monkeypatch.delenv("CASH_TRACE", raising=False)
    name = "untraced_impl_" + "".join(ch if ch.isalnum() else "_" for ch in request.node.name)
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(_IMPL), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    before = shadow.SHADOW
    shadow.disable()
    on = shadow.enable()
    assert on is not None and on.ok
    yield on, name
    shadow.disable()
    sys.modules.pop(name, None)
    if before is not None:
        shadow.enable()


def _run(processor, code):
    return processor.process_statement(code, annotation=_PERSIST)


def _stored_without_traces(on, monkeypatch):
    """Store as a version that kept no trace with the entry did."""
    monkeypatch.setattr(on, "statement_trace", lambda run: None)
    monkeypatch.setattr(on, "call_trace", lambda key, site, unit: None)


def test_an_entry_with_no_trace_runs_and_then_has_one(trace_on, statement_processor, mock_shell, monkeypatch):
    on, _ = trace_on
    _run(statement_processor, "def inner(x):\n    return x * 2")
    with monkeypatch.context() as m:
        _stored_without_traces(on, m)
        assert _run(statement_processor, "t = inner(3)")["status"] == CacheStatus.COMPUTED
    mock_shell.user_ns["inner"].__code__ = (lambda x: x * 10).__code__

    again = _run(statement_processor, "t = inner(3)")

    assert again["status"] == CacheStatus.COMPUTED, dict(on.counts)
    assert mock_shell.user_ns["t"] == 30
    assert on.counts["stmt:miss:found_not_served"] == 1, dict(on.counts)
    # Its run stored a trace: the next lookup is checked and served.
    assert _run(statement_processor, "t = inner(3)")["status"] == CacheStatus.RESTORED
    assert on.counts["stmt:hit:agree"] == 1, dict(on.counts)


def test_a_cached_call_with_no_trace_runs_again(trace_on, statement_processor, mock_shell, monkeypatch):
    on, mod = trace_on
    _run(statement_processor, f"import {mod}\nfrom {mod} import slow_inner")
    with monkeypatch.context() as m:
        _stored_without_traces(on, m)
        for _ in range(2):
            statement_processor.process_statement("x = slow_inner(2) + 0")
    assert mock_shell.user_ns["x"] == 4
    sys.modules[mod].inner.__code__ = (lambda x: x * 10).__code__

    statement_processor.process_statement("x = slow_inner(2) + 1")

    assert mock_shell.user_ns["x"] == 21, dict(on.counts)


@pytest.mark.parametrize("call", ["slow_inner(2, **kw)", "slow_inner(*args)"])
def test_a_cached_call_written_with_unpacking_is_checked_by_its_trace(trace_on, statement_processor, mock_shell, call):
    """A call with ``*``/``**`` is keyed on what it received: its key went
    unnoted, its hit unchecked, and ``inner`` edited inside it was served
    (5 where 21 is right)."""
    on, mod = trace_on
    _run(statement_processor, f"import {mod}\nfrom {mod} import slow_inner\nkw = {{'b': 0}}\nargs = (2,)")
    for _ in range(2):
        statement_processor.process_statement(f"y = {call} + 0")
    assert mock_shell.user_ns["y"] == 4
    sys.modules[mod].inner.__code__ = (lambda x: x * 10).__code__

    statement_processor.process_statement(f"y = {call} + 1")

    assert mock_shell.user_ns["y"] == 21, dict(on.counts)


def test_a_statement_above_is_restored_only_when_its_trace_vouches(trace_on, cash_magics):
    """A jump run restores a statement above the one run without running it
    (``try_virtual_restore``): that never asked the trace, and ``t`` came
    back 6 after ``inner`` was edited to ``x * 10``."""
    from tests._cell_driver import run_cash_cell
    from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

    cells = [
        "import time\ndef inner(x):\n    return x * 2",
        f"t = (inner(3), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]",
        "u = t + 1",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    assert (ns["t"], ns["u"]) == (6, 7)
    ns["inner"].__code__ = (lambda x: x * 10).__code__
    del ns["t"]

    run_cash_cell(cash_magics, cells[2], cells=cells)

    # ``u`` keeps the lineage ``t`` was restored with: a rerun forced by a
    # trace does not change it yet (the restart work's lineage fix).
    assert ns["t"] == 30


def test_a_statement_above_using_a_module_edited_before_it_loaded_is_not_restored(
    trace_on, cash_magics, tmp_path, monkeypatch
):
    """``g`` imports a module of the user's in its body. The module was
    edited while it was not loaded (a restart): nothing bound tells, but its
    file does. Restored, ``t`` came back 6."""
    from tests._cell_driver import run_cash_cell
    from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

    _, mod = trace_on
    name = mod + "_lazy"
    (tmp_path / f"{name}.py").write_text("def f(x):\n    return x * 2\n", encoding="utf-8")
    cells = [
        f"import time\ndef g(x):\n    import {name}\n    return {name}.f(x)",
        f"t = (g(3), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]",
        "u = t + 1",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    assert ns["t"] == 6
    (tmp_path / f"{name}.py").write_text("def f(x):\n    return x * 10\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, name)
    del ns["t"]

    run_cash_cell(cash_magics, cells[2], cells=cells)

    assert ns["t"] == 30
    sys.modules.pop(name, None)


def test_a_statement_above_reading_a_module_setting_changed_since_is_not_restored(trace_on, cash_magics, tmp_path):
    """``helpers.CFG = 5`` in a cell above, then a jump run: the restore
    skipped every global its trace names, a loaded module's too, and ``t``
    came back 4."""
    from tests._cell_driver import run_cash_cell
    from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

    _, mod = trace_on
    name = mod + "_cfg"
    (tmp_path / f"{name}.py").write_text("CFG = 1\ndef f(x):\n    return x + CFG\n", encoding="utf-8")
    cells = [
        f"import time, {name}",
        f"{name}.CFG = 1",
        f"t = ({name}.f(3), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]",
        "u = t + 1",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    assert ns["t"] == 4
    cells[1] = f"{name}.CFG = 5"
    run_cash_cell(cash_magics, cells[1], cells=cells)
    del ns["t"]

    run_cash_cell(cash_magics, cells[3], cells=cells)

    assert ns["t"] == 8
    sys.modules.pop(name, None)


def _jump_after_a_restart(cash_magics, cells, folder, name, edit=None, on_path=True):
    """Run *cells*, change module *name*'s file in *folder* (*edit*: its new
    text, None to delete it) while it is not loaded, as across a restart,
    then run the last cell alone."""
    from tests._cell_driver import run_cash_cell

    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    first = ns["t"]
    path = folder / f"{name}.py"
    if edit is None:
        path.unlink()
    elif edit is not False:
        path.write_text(edit, encoding="utf-8")
    sys.modules.pop(name, None)
    if not on_path:
        sys.path.remove(str(folder))
    del ns["t"]
    run_cash_cell(cash_magics, cells[-1], cells=cells)
    return first, ns


def _lazy_cells(name):
    from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

    return [
        f"import time\ndef g(x):\n    import {name}\n    return {name}.f(x)",
        f"t = (g(3), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]",
        "u = t + 1",
    ]


def test_a_module_found_through_a_path_a_cell_adds_is_read_from_its_file(trace_on, cash_magics, tmp_path):
    """``sys.path.insert`` in a cell above, which a jump run does not run:
    the module was not found, counted as a library's, and ``t`` came back
    6 after its file was edited to ``x * 10``. Its file is read where it
    was loaded from: not restored, ``t`` runs again, and without the cell
    above that cannot find the module, which cash says."""
    from cash.exceptions import UpstreamStateError

    _, mod = trace_on
    name = mod + "_src"
    folder = tmp_path / "src"
    folder.mkdir()
    (folder / f"{name}.py").write_text("def f(x):\n    return x * 2\n", encoding="utf-8")
    cells = [f"import sys\nsys.path.insert(0, {str(folder)!r})", *_lazy_cells(name)]

    with pytest.raises(UpstreamStateError, match="ModuleNotFoundError"):
        _jump_after_a_restart(cash_magics, cells, folder, name, "def f(x):\n    return x * 10\n", on_path=False)

    assert "t" not in cash_magics.shell.user_ns


def test_an_edited_default_of_a_module_not_loaded_is_seen(trace_on, cash_magics, tmp_path):
    """Defaults are not in a function's code: ``k=2`` edited to ``k=10``
    was restored as 6."""
    _, mod = trace_on
    name = mod + "_k"
    (tmp_path / f"{name}.py").write_text("def f(x, k=2):\n    return x * k\n", encoding="utf-8")

    first, ns = _jump_after_a_restart(
        cash_magics, _lazy_cells(name), tmp_path, name, "def f(x, k=10):\n    return x * k\n"
    )

    assert (first, ns["t"]) == (6, 30)


def test_a_statement_above_using_a_module_since_deleted_is_not_restored(trace_on, cash_magics, tmp_path):
    """Its module gone, the statement cannot be run again: it was restored
    as 6, a value the notebook no longer makes. It runs, and cash says it
    failed."""
    from cash.exceptions import UpstreamStateError

    on, mod = trace_on
    name = mod + "_gone"
    (tmp_path / f"{name}.py").write_text("def f(x):\n    return x * 2\n", encoding="utf-8")

    with pytest.raises(UpstreamStateError, match="ModuleNotFoundError"):
        _jump_after_a_restart(cash_magics, _lazy_cells(name), tmp_path, name, None)

    assert "t" not in cash_magics.shell.user_ns, dict(on.counts)


def test_a_module_not_loaded_and_unchanged_vouches_for_what_its_text_settles(trace_on, cash_magics, tmp_path):
    """Its functions calling each other (``helpers.step`` calls ``h0``):
    the global ``h0`` of a module not loaded could not be told unrun, and
    every such restore was refused, 2-2.7x slower jump runs. Its file
    unchanged settles it: a ``def`` once, a constant by value, a builtin
    it never binds."""
    on, mod = trace_on
    name = mod + "_steps"
    text = "SCALE = 2\ndef h0(x):\n    return x * SCALE\ndef f(x):\n    return len([h0(x)]) * h0(x)\n"
    (tmp_path / f"{name}.py").write_text(text, encoding="utf-8")

    first, ns = _jump_after_a_restart(cash_magics, _lazy_cells(name), tmp_path, name, False)

    assert (first, ns["t"]) == (6, 6)
    assert on.counts["upstream:hit:agree"] >= 1, dict(on.counts)
    assert not [k for k in on.counts if k.startswith("upstream:miss")], dict(on.counts)


def test_a_constant_of_a_module_not_loaded_is_compared_by_value(trace_on, cash_magics, tmp_path):
    """``SCALE = 2`` edited to ``SCALE = 10`` is a change of the file, and
    of the constant: not restored."""
    _, mod = trace_on
    name = mod + "_scale"
    (tmp_path / f"{name}.py").write_text("SCALE = 2\ndef f(x):\n    return x * SCALE\n", encoding="utf-8")

    first, ns = _jump_after_a_restart(
        cash_magics, _lazy_cells(name), tmp_path, name, "SCALE = 10\ndef f(x):\n    return x * SCALE\n"
    )

    assert (first, ns["t"]) == (6, 30)


def test_a_module_setting_read_through_a_local_import_and_set_above_is_seen(trace_on, cash_magics, tmp_path):
    """``g`` imports the module in its body and reads ``CFG``: the trace
    names nothing of it, and the restore's key kept the value recorded
    before ``CFG = 5`` in a cell above, so ``t`` came back 4."""
    from tests._cell_driver import run_cash_cell
    from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

    _, mod = trace_on
    name = mod + "_lcfg"
    (tmp_path / f"{name}.py").write_text("CFG = 1\n", encoding="utf-8")
    cells = [
        f"import time, {name}",
        f"{name}.CFG = 1",
        f"def g(x):\n    import {name}\n    return x + {name}.CFG",
        f"t = (g(3), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]",
        "u = t + 1",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    assert ns["t"] == 4
    cells[1] = f"{name}.CFG = 5"
    run_cash_cell(cash_magics, cells[1], cells=cells)
    del ns["t"]

    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert ns["t"] == 8
    sys.modules.pop(name, None)


def test_a_check_that_could_not_start_does_not_serve(trace_on, statement_processor, mock_shell, monkeypatch):
    on, _ = trace_on
    _run(statement_processor, "def inner(x):\n    return x * 2")
    _run(statement_processor, "t = inner(3)")
    assert _run(statement_processor, "t = inner(3)")["status"] == CacheStatus.RESTORED

    def broken(*args, **kwargs):
        raise OSError("the cache folder's mount is not reachable")

    monkeypatch.setattr(on, "attach", broken)
    mock_shell.user_ns["inner"].__code__ = (lambda x: x * 10).__code__

    assert _run(statement_processor, "t = inner(3)")["status"] == CacheStatus.COMPUTED
    assert mock_shell.user_ns["t"] == 30
    assert on.counts["errors"] >= 1


def test_with_no_tool_to_record_traces_the_key_decides(trace_on, statement_processor, monkeypatch):
    on, _ = trace_on
    _run(statement_processor, "def inner(x):\n    return x * 2")
    _stored_without_traces(on, monkeypatch)
    _run(statement_processor, "t = inner(3)")
    monkeypatch.setattr(on, "ok", False)

    assert _run(statement_processor, "t = inner(3)")["status"] == CacheStatus.RESTORED


class _FlakyFolder(FileBackend):
    @property
    def local_dir(self):
        raise OSError("the cache folder's mount is not reachable")


def test_a_local_dir_that_raises_keeps_traces_in_the_temp_dir(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("CASH_CACHE_DIR", raising=False)
    kept = shadow.Shadow()

    kept.attach({"__name__": "__main__"}, {}, _FlakyFolder(cache_dir=str(tmp_path)))

    assert kept.dir == os.path.join(tempfile.gettempdir(), "shadow")
    assert kept.counts["errors"] == 1
    assert "local_dir failed" in caplog.text


def _edit_later(path, text):
    """Write *text* to *path* with a later mtime, so Python's own compiled
    file check sees the edit however quick (same size, same second)."""
    st = path.stat()
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


@pytest.mark.parametrize(
    ("body", "change"),
    [
        ("import {s}\nTABLE = {{'k': 2}}\nTABLE.update({s}.OVERRIDES)\n", "settings"),
        ("import os\nTABLE = {{'k': 2}}\nif os.environ.get({var!r}):\n    TABLE['k'] = 9\n", "environ"),
    ],
    ids=["updated_from_another_module", "set_by_an_environment_variable"],
)
def test_a_module_body_that_runs_code_vouches_for_none_of_its_settings(
    trace_on, cash_magics, tmp_path, monkeypatch, body, change
):
    """``TABLE = {'k': 2}`` then ``TABLE.update(settings.OVERRIDES)``, or a
    branch on the environment: the file unchanged, its text does not settle
    ``TABLE``, which was vouched for and served as 6 after the other module
    or the environment changed."""
    _, mod = trace_on
    name, settings, var = mod + "_tab", mod + "_set", (mod + "_BIG").upper()
    (tmp_path / f"{settings}.py").write_text("OVERRIDES = {}\n", encoding="utf-8")
    text = body.format(s=settings, var=var) + "def f(x):\n    return x * TABLE['k']\n"
    (tmp_path / f"{name}.py").write_text(text, encoding="utf-8")
    from tests._cell_driver import run_cash_cell

    cells = _lazy_cells(name)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    assert ns["t"] == 6
    if change == "settings":
        _edit_later(tmp_path / f"{settings}.py", "OVERRIDES = {'k': 9}\n")
    else:
        monkeypatch.setenv(var, "1")
    sys.modules.pop(name, None)
    sys.modules.pop(settings, None)
    del ns["t"]
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert ns["t"] == 27


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (
            "class C:\n    K = 2\ndef f(x):\n    return x * C.K\n",
            "class C:\n    K = 9\ndef f(x):\n    return x * C.K\n",
        ),
        ("def f(x, k=2):\n    return x * k\n", "def f(x, k=9):\n    return x * k\n"),
    ],
    ids=["class_attribute", "default"],
)
def test_a_file_edited_while_its_module_is_loaded_does_not_vouch_for_it(trace_on, cash_magics, tmp_path, before, after):
    """The module loaded, its file edited, a statement then stored: the
    trace held what the old module computed and the new file's digest, and
    after a restart the unchanged file vouched for 6. What was loaded is
    what the file is held against."""
    from tests._cell_driver import run_cash_cell

    _, mod = trace_on
    name = mod + "_lg"
    path = tmp_path / f"{name}.py"
    path.write_text(before, encoding="utf-8")
    cells = [*_lazy_cells(name)]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    run_cash_cell(cash_magics, "w = g(1)", cells=cells)
    _edit_later(path, after)
    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = cash_magics.shell.user_ns
    assert ns["t"] == 6  # the module loaded is the old one
    sys.modules.pop(name, None)
    del ns["t"]
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert ns["t"] == 27


def test_a_file_swapped_keeping_its_mtime_and_size_is_read_again(tmp_path):
    """``cp -p`` or ``touch -r``: the same mtime and size, other text. The
    file's state was kept by mtime and size and the old one served."""
    from cash.notebook import shadow

    path = tmp_path / "swapped.py"
    path.write_text("def f(x):\n    return x * 2\n", encoding="utf-8")
    old = path.stat()
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns - 10_000_000_000))
    old = path.stat()
    first = shadow._file_state(str(path))
    path.write_text("def f(x):\n    return x * 9\n", encoding="utf-8")
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))

    assert shadow._file_state(str(path)).digest != first.digest


def test_a_docstring_edit_changes_the_files_digest(tmp_path):
    """A docstring can be read as the code runs (``int(f.__doc__)``,
    ``description=__doc__``): an edit of one is an edit of the file."""
    from cash.notebook import shadow

    path = tmp_path / "doc.py"
    path.write_text('def f(x):\n    "2"\n    return x\n', encoding="utf-8")
    first = shadow._file_state(str(path)).digest
    path.write_text('def f(x):\n    "9"\n    return x\n# a comment\n', encoding="utf-8")

    assert shadow._file_state(str(path)).digest != first

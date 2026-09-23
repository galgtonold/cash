"""A cell's file bookkeeping digests each input file about once, not once per use.

Counted, not timed: content digests taken in the live kernel while one cell
runs (``hashlib.sha256`` inside ``file_dep_snapshot``, so a memo hit does not
count). A profile of real notebooks put most of cash's notebook overhead
here:

* One notebook read a folder in a loop, ``d = pd.read_csv(f)``. ``d``'s recorded files
  were merged across iterations, so iteration k snapshotted all k files so far:
  865,265 hashes for 1,312 files in one cell.
* Another read 5,222 files into ``docs``; each statement derived from it
  re-snapshotted all 5,222 when it was saved, and re-checked them all again in
  the upstream simulation.

A relative read is recorded under two spellings (the resolved path, and the
relative one so an ``os.chdir`` is seen), so these fixtures carry 2N paths for
N files -- one digest serves both.
"""

import ast
import os
from pathlib import Path

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

N = 80

#: Counts lookups (``file_content_hash`` calls) and digests (``sha256``) per
#: cell run, and closes the 5 s reuse window: 80 small files are checked in
#: well under five seconds, where 5,222 were not -- without this the
#: fixture is served from the window and passes on code that re-hashed 5,222 files
#: for minutes.
#:
#: Per cell run, because the claim is per cell run -- a new one looks at the
#: files again, once -- and because a peek is an execution too: the first one
#: after a cell runs the upstream check over that cell's new entries, before
#: its expression is evaluated. (Before cell runs were numbered there is one
#: bucket, ``None``, and it holds everything.)
#:
#: The counter wraps the ORIGINALS, kept on the module, and `_counting` puts
#: them back: a worker reuses its kernel for the next test, module state and
#: all, and a second test's counter stacked on the first counted every call
#: twice -- which read as the check running twice.
_M = "__import__('cash.notebook.file_dep_snapshot', fromlist=['_'])"
_BUMP = (
    "m._test_n.setdefault(getattr(m, '_HASH_EPOCH', None), [0, 0]).__setitem__({i}, "
    "m._test_n[getattr(m, '_HASH_EPOCH', None)][{i}] + 1)"
)
COUNTER = (
    "(lambda m, types: (lambda s: (setattr(m, '_test_saved', s), setattr(m, '_test_n', {}),"
    " setattr(m, '_HASH_MEMO_TTL_SECONDS', 0.0),"
    " setattr(m, 'hashlib', types.SimpleNamespace(sha256=(lambda f: (lambda *a: "
    f"({_BUMP.format(i=1)}, f(*a))[1]))(s[0].sha256))),"
    " setattr(m, 'file_content_hash', (lambda f: (lambda *a, **k: "
    f"({_BUMP.format(i=0)}, f(*a, **k))[1]))(s[1]))))"
    "(getattr(m, '_test_saved', None) or (m.hashlib, m.file_content_hash, m._HASH_MEMO_TTL_SECONDS)))"
    f"({_M}, __import__('types'))"
)
UNINSTALL = (
    "(lambda m: (lambda s: s and (setattr(m, 'hashlib', s[0]), setattr(m, 'file_content_hash', s[1]),"
    " setattr(m, '_HASH_MEMO_TTL_SECONDS', s[2]), delattr(m, '_test_saved')))"
    f"(getattr(m, '_test_saved', None)))({_M})"
)
RESET = f"{_M}._test_n.clear()"
READ = f"{_M}._test_n"


@pytest.fixture
def _counting(nb_runner):
    yield
    try:
        nb_runner.peek(UNINSTALL)
    except Exception:  # noqa: BLE001 - a kernel that never started has nothing to undo
        pass


SETUP = "import glob\nimport os\nimport pandas as pd\nfiles = sorted(glob.glob('exports/*.csv'))"
LOOP = (
    "parts = []\n"
    "for f in files:\n"
    "    d = pd.read_csv(f)\n"
    "    d['source_file'] = os.path.basename(f)\n"
    "    parts.append(d)\n"
    "raw = pd.concat(parts, ignore_index=True)\n"
    "print('rows', len(raw))"
)
COMPREHENSION = "raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)\nprint('rows', len(raw))"
DERIVED = "\n".join(f"s{k} = raw[raw['v'] > {k}].copy()" for k in range(6)) + "\nprint(len(s5))"


def _files(work):
    folder = Path(work) / "exports"
    folder.mkdir()
    for i in range(N):
        path = folder / f"e{i:03d}.csv"
        path.write_text("g,v\n" + "\n".join(f"{i},{j}" for j in range(40)) + "\n")
        st = os.stat(path)
        os.utime(path, (st.st_atime - 3600, st.st_mtime - 3600))


def _counts_for(nb_runner, cell):
    """The most lookups and the most digests any one cell run took, over the
    cell and the check after it."""
    nb_runner.peek(RESET)
    nb_runner.run_cell(cell)
    table = ast.literal_eval(nb_runner.peek(READ))
    print(f"cell {cell}: {{run: [lookups, digests]}} = {table}")
    # Nothing counted at all is the best case, not a broken counter: inputs
    # that are settled and unchanged are not read (`_unchanged_since_hashed`).
    # The loop test's cell reads its files, so there the counter does count.
    return (max((v[0] for v in table.values()), default=0), max((v[1] for v in table.values()), default=0))


def test_a_loop_over_files_digests_each_file_once(nb_runner, _counting):
    _files(nb_runner.work_dir)
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, LOOP])
    nb_runner.start_kernel()
    nb_runner.peek(COUNTER)
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    lookups, digests = _counts_for(nb_runner, 3)
    assert f"rows {N * 40}" in nb_runner.get_output(3)
    assert digests <= N + 10, f"{digests} digests of {N} files in one cell run"
    # Measured 480 in the cell; 7,120 when the loop's bookkeeping grew with N squared.
    assert lookups <= 10 * N, f"{lookups} lookups for {N} files: the loop's bookkeeping grows with N squared"


def test_statements_derived_from_many_files_do_not_redigest_them(nb_runner, _counting):
    _files(nb_runner.work_dir)
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, COMPREHENSION, DERIVED])
    nb_runner.start_kernel()
    nb_runner.peek(COUNTER)
    nb_runner.run_cell(1)
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    lookups, digests = _counts_for(nb_runner, 4)
    assert str(N * 34) in nb_runner.get_output(4)
    assert digests <= N + 10, f"{digests} digests of {N} inherited files in one cell run"
    # The check after the cell answers each file once per run: 160 lookups (2
    # spellings x 80 files), where each of the 6 new entries used to check all
    # of them (960) and saving them did too (2,094).
    assert lookups <= 4 * N, f"{lookups} lookups for {N} inherited files across 6 derived statements"

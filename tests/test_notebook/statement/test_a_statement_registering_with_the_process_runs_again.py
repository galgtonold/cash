"""A statement that registers something with the process runs again.

``atexit.register(flush)``, ``signal.signal(SIGUSR1, on_usr1)``,
``copyreg.pickle(K, reduce_k)``, ``logging.getLogger("lib").setLevel(5)``:
each changes a registry the process keeps outside any variable. Restored
from the cache, the statement did not change it again: after a restart
(here, the registration undone as a new process would not have it) the
handler was never installed. Each such statement, written in the cell or
in a function it calls, is not stored (`cash.tracking.process_registries`).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings

import pytest

from tests._cell_driver import run_cash_cell

SETUP = (
    "import atexit, codecs, copyreg, decimal, logging, os, signal, sys, threading, warnings\n"
    "def cb(*args, **kwargs):\n    return None\n"
    "class K:\n    pass\n"
    "class KWarning(Warning):\n    pass\n"
    "def reduce_k(k):\n    return (K, ())\n"
    "LOOKED = []\n"
    "def search(name):\n    LOOKED.append(name)\n    return None\n"
    "HANDLER = logging.NullHandler()\n"
    "class Finder:\n    def find_spec(self, *args):\n        return None\n"
    "FINDER = Finder()\n"
    "OLD_DISPLAYHOOK = sys.displayhook\n"
    "import random\nrandom.seed(5)\nFIRST = random.random()\nrandom.seed(None)\n"
    "try:\n    import numpy as np\n    np.random.seed(5)\n    FIRST_NP = np.random.random()\n"
    "    np.random.seed(None)\nexcept ImportError:\n    pass\n"
    "try:\n    import matplotlib\n    import matplotlib.style\nexcept ImportError:\n    pass\n"
    "OLD_LIMIT = sys.getrecursionlimit()\n"
    "try:\n    from cash.tracking.process_registries import _OWN_ATEXIT\n"
    "except ImportError:  # before it\n    _OWN_ATEXIT = [0]\n"
    "def users_atexit():\n"
    "    return atexit._ncallbacks() - _OWN_ATEXIT[0]\n"
)


#: name -> (statement, whether its effect is there, undo it as a new
#: process would not have it).
CASES = {
    # ``atexit._ncallbacks()`` counts what was ever registered, cash's
    # own apart (`register_atexit`): the statement adds one each run.
    "atexit": ("atexit.register(cb)", "users_atexit() > COUNT", "atexit.unregister(cb); COUNT = users_atexit()"),
    "signal": (
        "signal.signal(signal.SIGUSR1, cb)",
        "signal.getsignal(signal.SIGUSR1) is cb",
        "signal.signal(signal.SIGUSR1, signal.SIG_DFL)",
    ),
    "excepthook": ("sys.excepthook = cb", "sys.excepthook is cb", "sys.excepthook = sys.__excepthook__"),
    "threading-excepthook": (
        "threading.excepthook = cb",
        "threading.excepthook is cb",
        "threading.excepthook = threading.__excepthook__",
    ),
    "displayhook": ("sys.displayhook = cb", "sys.displayhook is cb", "sys.displayhook = OLD_DISPLAYHOOK"),
    "setprofile": ("sys.setprofile(cb)", "sys.getprofile() is cb", "sys.setprofile(None)"),
    "recursionlimit": (
        "sys.setrecursionlimit(4321)",
        "sys.getrecursionlimit() == 4321",
        "sys.setrecursionlimit(OLD_LIMIT)",
    ),
    "simplefilter": (
        "warnings.simplefilter('error', category=KWarning)",
        "any(f[2] is KWarning for f in warnings.filters)",
        "warnings.filters[:] = [f for f in warnings.filters if f[2] is not KWarning]; warnings._filters_mutated()",
    ),
    "logger-level": (
        "logging.getLogger('zz.lib').setLevel(5)",
        "logging.getLogger('zz.lib').level == 5",
        "logging.getLogger('zz.lib').setLevel(0)",
    ),
    "logger-handler": (
        "logging.getLogger('zz.lib2').addHandler(HANDLER)",
        "HANDLER in logging.getLogger('zz.lib2').handlers",
        "logging.getLogger('zz.lib2').removeHandler(HANDLER)",
    ),
    "copyreg": (
        "copyreg.pickle(K, reduce_k)",
        "copyreg.dispatch_table.get(K) is reduce_k",
        "del copyreg.dispatch_table[K]",
    ),
    "codecs": (
        "codecs.register(search)",
        "",  # `_codec_found`
        "codecs.unregister(search)",
    ),
    "sys-path": ("sys.path.append('/zz-probe')", "'/zz-probe' in sys.path", "sys.path.remove('/zz-probe')"),
    "meta-path": ("sys.meta_path.append(FINDER)", "FINDER in sys.meta_path", "sys.meta_path.remove(FINDER)"),
    "sys-modules": ("sys.modules['zz_probe'] = K()", "'zz_probe' in sys.modules", "del sys.modules['zz_probe']"),
    "environ": ("os.environ['ZZ_PROBE'] = '1'", "os.environ.get('ZZ_PROBE') == '1'", "del os.environ['ZZ_PROBE']"),
    "rcParams": (
        "matplotlib.rcParams['lines.linewidth'] = 7.5",
        "matplotlib.rcParams['lines.linewidth'] == 7.5",
        "matplotlib.rcParams['lines.linewidth'] = 1.5",
    ),
    "rc-function": (
        "matplotlib.rc('lines', linewidth=7.5)",
        "matplotlib.rcParams['lines.linewidth'] == 7.5",
        "matplotlib.rcParams['lines.linewidth'] = 1.5",
    ),
    "style": (
        "matplotlib.style.use('ggplot')",
        "matplotlib.rcParams['axes.facecolor'] == '#E5E5E5'",
        "matplotlib.rcParams['axes.facecolor'] = 'white'",
    ),
    "np-seterr": ("np.seterr(divide='raise')", "np.geterr()['divide'] == 'raise'", "np.seterr(divide='warn')"),
    # Drawn and seeded streams are put back with the entry: restored, not run.
    "np-random-seed": ("np.random.seed(5)", "np.random.random() == FIRST_NP", "np.random.seed(None)"),
    "random-seed": ("random.seed(5)", "random.random() == FIRST", "random.seed(None)"),
    "decimal": ("decimal.getcontext().prec = 7", "decimal.getcontext().prec == 7", "decimal.getcontext().prec = 28"),
}

#: Found wrong before (restored, the effect missing), written in the cell.
#: The others the statement's own text already kept out of the cache.
IN_THE_CELL = (
    "rcParams",
    "rc-function",
    "style",
    "np-seterr",
    "np-random-seed",
    "random-seed",
    "atexit",
    "signal",
    "setprofile",
    "recursionlimit",
    "logger-level",
    "logger-handler",
    "copyreg",
    "codecs",
)


def _codec_found(ns) -> bool:
    import codecs

    ns["LOOKED"].clear()
    try:
        codecs.lookup("zz-probe-codec")
    except LookupError:
        pass
    return "zz_probe_codec" in ns["LOOKED"] or "zz-probe-codec" in ns["LOOKED"]


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


def _params():
    for kind in CASES:
        if kind in IN_THE_CELL:
            yield pytest.param(kind, False, id=f"{kind}-in-the-cell")
        yield pytest.param(kind, True, id=f"{kind}-in-a-function")


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize(("kind", "in_a_function"), list(_params()))
def test_the_registration_is_made_again(kind, in_a_function, cash_magics, mock_shell):
    if kind in ("rcParams", "rc-function", "style"):
        pytest.importorskip("matplotlib")
    elif kind.startswith("np-"):
        pytest.importorskip("numpy")
    statement, there, undo = CASES[kind]
    setup = SETUP
    if in_a_function:
        setup += "def register():\n" + textwrap.indent(statement, "    ") + "\n"
        statement = "register()"
    cells = [setup, statement]
    ns = mock_shell.user_ns

    def found() -> bool:
        return _codec_found(ns) if kind == "codecs" else eval(there, ns)

    run_cash_cell(cash_magics, setup, cells=cells)
    ns["COUNT"] = eval("users_atexit()", ns)
    try:
        run_cash_cell(cash_magics, statement, cells=cells)
        assert found()
        exec(undo, ns)
        assert not found()
        run_cash_cell(cash_magics, statement, cells=cells)
        statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
        assert found(), f"restored without registering again: {statuses}"
    finally:
        try:
            exec(undo, ns)
        except Exception:
            pass


@pytest.mark.usefixtures("store_everything")
def test_a_statement_that_registers_nothing_is_still_stored(cash_magics):
    cells = [SETUP, "total = sum(range(10))\nlogging.getLogger('zz.made').debug('made')"]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert statuses[0] == "RESTORED", statuses


@pytest.mark.usefixtures("store_everything")
def test_a_statement_registering_every_time_is_warned_about_the_second_time(cash_magics):
    """Not the first time: a library setting itself up once a process on
    first use (tqdm's monitor) registers only then, and is stored next."""
    statement = "atexit.register(cb)"
    cells = [SETUP, statement]
    run_cash_cell(cash_magics, SETUP, cells=cells)
    codes = []
    try:
        for _ in range(2):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                run_cash_cell(cash_magics, statement, cells=cells)
            codes.append([getattr(w.message, "code", None) for w in caught])
    finally:
        import atexit

        atexit.unregister(cash_magics.shell.user_ns["cb"])
    assert "NOTEBOOK-UNSEEN-WRITES" not in codes[0]
    assert "NOTEBOOK-UNSEEN-WRITES" in codes[1]


#: A setting the next run makes again, where it already is: no change seen
#: then, and stored, it was served after a restart without being made.
SAME_AGAIN = {
    "logger-level": (
        "logging.getLogger('zz.again').setLevel(10)",
        "logging.getLogger('zz.again').level == 10",
        "logging.getLogger('zz.again').setLevel(0)",
    ),
    "disable": ("logging.disable(logging.INFO)", "logging.root.manager.disable == 20", "logging.disable(0)"),
    "recursionlimit": (
        "sys.setrecursionlimit(4321)",
        "sys.getrecursionlimit() == 4321",
        "sys.setrecursionlimit(OLD_LIMIT)",
    ),
}


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize("kind", list(SAME_AGAIN))
def test_a_setting_made_again_where_it_already_is_is_made_after_a_restart(kind, cash_magics, mock_shell):
    statement, there, undo = SAME_AGAIN[kind]
    cells = [SETUP, statement]
    ns = mock_shell.user_ns
    run_cash_cell(cash_magics, SETUP, cells=cells)
    try:
        for _ in range(3):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
        # A restart: the session forgotten, the setting as a new process finds it.
        cash_magics.tracking_state.reset_session_state()
        cash_magics._upstream_checker.simulator.cache.reset()
        exec(undo, ns)
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
        assert eval(there, ns), f"restored without making the setting: {statuses}"
    finally:
        exec(undo, ns)


@pytest.mark.usefixtures("store_everything")
def test_a_registration_made_only_the_first_time_still_runs_again(cash_magics):
    """A library's setup made once a process on first use (tqdm's monitor):
    a new process makes it again, so a served entry would skip it."""
    statement = "ONCE = ONCE if 'ONCE' in globals() else atexit.register(cb)\ntotal = sum(range(10))"
    cells = [SETUP, statement]
    run_cash_cell(cash_magics, SETUP, cells=cells)
    try:
        for _ in range(3):
            run_cash_cell(cash_magics, statement, cells=cells)
    finally:
        import atexit

        atexit.unregister(cash_magics.shell.user_ns["cb"])
    statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert statuses[0] != "RESTORED", statuses


def test_an_audit_hook_added_is_seen():
    """``sys.addaudithook`` cannot be undone, so in a process of its own."""
    script = textwrap.dedent(
        """
        import sys
        from cash.tracking import process_registries
        before = process_registries.snapshot()
        print(process_registries.changed(before))
        sys.addaudithook(lambda event, args: None)
        print(process_registries.changed(before))
        """
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=True)
    first, second = done.stdout.strip().splitlines()[-2:]
    assert first == "[]"
    assert "sys.addaudithook" in second


def test_an_import_while_another_thread_imports_is_still_the_import_s(tmp_path):
    """What a module sets on its first import is the module's, also when
    another thread was in the middle of an import of its own as it began
    and has ended since: the depth of imports is each thread's."""
    import importlib
    import logging
    import threading

    from cash.tracking import process_registries

    (tmp_path / "zz_gate.py").write_text(
        "import threading\nSTARTED = threading.Event()\nGO = threading.Event()\nDONE = threading.Event()\n",
        encoding="utf-8",
    )
    (tmp_path / "zz_slow_import.py").write_text(
        "import zz_gate\nzz_gate.STARTED.set()\nzz_gate.GO.wait(30)\n", encoding="utf-8"
    )
    (tmp_path / "zz_sets_a_level.py").write_text(
        "import logging, zz_gate\nzz_gate.GO.set()\nzz_gate.DONE.wait(30)\nlogging.getLogger('zz.thread').setLevel(10)\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    gate = importlib.import_module("zz_gate")

    def other_imports():
        importlib.import_module("zz_slow_import")
        gate.DONE.set()

    other = threading.Thread(target=other_imports)
    try:
        before = process_registries.snapshot()
        other.start()
        assert gate.STARTED.wait(30)
        # Begun inside the other thread's import, ended after it.
        importlib.import_module("zz_sets_a_level")
        other.join(30)
        assert process_registries.changed(before) == []
    finally:
        gate.GO.set()
        gate.DONE.set()
        other.join(30)
        sys.path.remove(str(tmp_path))
        for name in ("zz_gate", "zz_slow_import", "zz_sets_a_level"):
            sys.modules.pop(name, None)
        logging.getLogger("zz.thread").setLevel(0)


def test_an_atexit_callback_taken_back_is_not_a_change():
    """``joblib.Parallel`` registers a clean-up with ``atexit`` for each
    run and unregisters it when the run is done: nothing is left, whether
    ``atexit``'s count of callbacks stays up (to Python 3.13) or goes down
    with it (3.14 on)."""
    import atexit

    from cash.tracking import process_registries

    def cleanup():
        return None

    before = process_registries.snapshot()
    atexit.register(cleanup)
    atexit.unregister(cleanup)
    assert process_registries.changed(before) == []
    before = process_registries.snapshot()
    atexit.register(cleanup)
    try:
        assert process_registries.changed(before) == ["atexit's callbacks (atexit.register)"]
    finally:
        atexit.unregister(cleanup)


def test_a_joblib_run_changes_no_registry():
    """joblib registers a clean-up of each temporary folder it makes and
    unregisters it once it deleted the folder. From Python 3.14 that
    unregister takes the callback out of ``atexit``'s count, which read as
    the run having changed ``atexit``'s callbacks. The library's own
    registrations are left out, and so are its own unregisters of them."""
    np = pytest.importorskip("numpy")
    joblib = pytest.importorskip("joblib")
    from cash.tracking import process_registries

    # Over joblib's 1 MB ``max_nbytes``: the array goes to such a folder.
    big = np.arange(300_000, dtype=float)
    for _ in range(2):
        before = process_registries.snapshot()
        sums = joblib.Parallel(n_jobs=2)(joblib.delayed(np.sum)(big) for _ in range(2))
        assert sums == [big.sum()] * 2
        assert process_registries.changed(before) == []


LDA_CELLS = [
    "import numpy as np\nfrom sklearn.decomposition import LatentDirichletAllocation\n"
    "X = np.random.default_rng(0).poisson(0.5, size=(200, 50))",
    "lda = LatentDirichletAllocation(n_components=3, max_iter=2, random_state=0, n_jobs=2).fit(X)",
]


@pytest.mark.usefixtures("store_everything")
@pytest.mark.timeout(120)
def test_a_fit_on_two_jobs_is_stored(cash_magics):
    """scikit-learn's ``n_jobs=2`` runs through joblib, whose ``atexit``
    clean-up removes a temporary folder only the run makes: a hit makes no
    folder, so skipping it changes nothing (p13's 36 s LDA fit ran again
    every time on Python 3.14)."""
    pytest.importorskip("sklearn")
    for _ in range(2):
        for cell in LDA_CELLS:
            run_cash_cell(cash_magics, cell, cells=LDA_CELLS)
    statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert statuses == ["RESTORED"], statuses


@pytest.mark.usefixtures("store_everything")
def test_a_library_s_callable_the_user_registers_is_registered_after_a_restart(cash_magics, mock_shell, tmp_path):
    """``atexit.register(df.to_csv, path)``, ``atexit.register(print, 'bye')``:
    a library's function, registered by the user's code, is the user's
    registration (the CSV is written at exit); only the registrations a
    library makes itself are left out."""
    import atexit

    pd = pytest.importorskip("pandas")
    out = tmp_path / "out.csv"
    statements = [f"atexit.register(df.to_csv, {str(out)!r})", "h = atexit.register(print, 'bye')"]
    cells = ["import atexit, pandas as pd\ndf = pd.DataFrame({'a': [1, 2]})", *statements]
    try:
        for _ in range(2):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
        cash_magics.tracking_state.reset_session_state()
        cash_magics._upstream_checker.simulator.cache.reset()
        before = atexit._ncallbacks()
        statuses = []
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
            if cell in statements:
                statuses.append(str(cash_magics._last_cell_metrics["statements"][0]["status"]))
        assert atexit._ncallbacks() - before == len(statements), statuses
    finally:
        atexit.unregister(print)
        df = mock_shell.user_ns.get("df")
        if isinstance(df, pd.DataFrame):
            atexit.unregister(df.to_csv)

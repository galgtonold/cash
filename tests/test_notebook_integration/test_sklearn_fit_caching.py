"""CAS-138 / CAS-137: how cash caches sklearn's estimator.fit().

Ground-truth probes for the design decision:
  * bare ``model.fit(X, y)``       -> caches + restores IN PLACE (CAS-138)
  * ``model = model.fit(X, y)``    -> caches (pure reassignment shape)
  * ``m = Model().fit(X, y)``      -> caches; restores across a real restart

The correctness crux (``test_bare_fit_alias_correctness``): a bare in-place fit
must restore by transferring the fitted state onto the EXISTING receiver object,
so aliases (``backup = clf``) see the fit -- a rebind of only ``clf`` would leave
``backup`` pointing at the stale, unfitted object.
"""
import asyncio

import pytest

pytest.importorskip("sklearn")

pytestmark = [pytest.mark.libraries, pytest.mark.timeout(180)]

SETUP = (
    "import numpy as np\n"
    "from sklearn.ensemble import RandomForestClassifier\n"
    "from sklearn.datasets import make_classification\n"
    "import cash\n"
    "%cash_on\n"
    "%cash_badge print"
)
DATA = "X, y = make_classification(n_samples=6000, n_features=20, random_state=0)"
MODEL = "clf = RandomForestClassifier(n_estimators=160, random_state=0)"


def _restart(nb_runner):
    """Restart the kernel in place and re-inject the notebook path."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(nb_runner.client.km._async_restart_kernel(now=True))
    loop.run_until_complete(nb_runner.client.kc._async_wait_for_ready(timeout=30))
    nb_runner._inject_notebook_path()


def test_bare_fit_now_caches(nb_runner):
    """CAS-138: bare ``clf.fit(X, y)`` now CACHES and restores on an isolated re-run.

    (Previously refused as an in-place mutation -- the most expensive cell in an
    ML notebook never cached. Mirrors ``test_reassign_fit_caches``: the receiver
    is self-referential (input + output), so the isolated re-run re-derives the
    unfitted estimator upstream and the fit cell lands a cache hit.)
    """
    nb_runner.create_notebook([
        SETUP, DATA, MODEL,
        "clf.fit(X, y)\nprint('fitted', clf.n_estimators)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.run_cell(4)  # isolated re-run should be a cache hit
    out = nb_runner.get_output(4)
    assert "RESTORED" in out, f"bare fit did not cache: {out!r}"


def test_reassign_fit_caches(nb_runner):
    """Form 2: ``clf = clf.fit(X, y)`` (add ``clf = ``) caches and restores on re-run."""
    nb_runner.create_notebook([
        SETUP, DATA, MODEL,
        "clf = clf.fit(X, y)\nprint('fitted', clf.n_estimators)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.run_cell(4)  # isolated re-run should be a cache hit
    out = nb_runner.get_output(4)
    assert "RESTORED" in out, f"reassign-fit did not cache: {out!r}"


def test_construct_fit_assign_restores_after_restart(nb_runner):
    """Form 3 + CAS-137: ``m = Model().fit(X, y)`` restores after a REAL restart.

    Forced to disk with ``# @cash:persist`` so the test probes the restore path
    itself, not the cost model's RAM-only decision for a cheap fit.
    """
    nb_runner.create_notebook([
        SETUP, DATA,
        "# @cash:persist\n"
        "m = RandomForestClassifier(n_estimators=160, random_state=0).fit(X, y)\n"
        "print('n =', m.n_estimators)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "n = 160" in nb_runner.get_output(3)

    _restart(nb_runner)
    nb_runner.run_cell(1)  # imports + %cash_on
    nb_runner.run_cell(2)  # X, y (re-derives the same lineage)

    nb_runner.run_cell(3)
    out = nb_runner.get_output(3)
    assert "RESTORED" in out, f"fitted estimator did not restore after restart (CAS-137): {out!r}"
    assert "n = 160" in out


def test_bare_fit_alias_correctness(nb_runner):
    """CAS-138 GATE: a bare-fit cache HIT restores IN PLACE so aliases see the fit.

    ``backup = clf`` aliases the estimator; the real ``clf.fit`` mutates the
    shared object so both observe it. A cache-hit REBIND of only ``clf`` (to a
    deep-copied cached estimator) would leave ``backup`` pointing at the stale,
    unfitted object -- the wrong-result bug this fix exists to prevent. The
    correct restore transfers the fitted state onto the existing receiver.

    Deterministic setup via a real restart: cells 1-4 run in order so ``backup``
    aliases the freshly rebuilt (unfitted) ``clf`` right before the fit cell,
    which then lands a persisted-cache HIT. The check cell is ``@cash:no-cache``
    so it re-reads the LIVE namespace instead of replaying its run_all output.
    """
    nb_runner.create_notebook([
        SETUP, DATA, MODEL,                       # 1, 2, 3
        "backup = clf",                           # 4
        "# @cash:persist\nclf.fit(X, y)",         # 5
        "# @cash:no-cache\n"                       # 6
        "print('clf_fit', hasattr(clf, 'classes_'))\n"
        "print('backup_fit', hasattr(backup, 'classes_'))\n"
        "print('same', clf is backup)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    _restart(nb_runner)
    nb_runner.run_cell(1)  # imports + %cash_on
    nb_runner.run_cell(2)  # X, y
    nb_runner.run_cell(3)  # clf (fresh, unfitted)
    nb_runner.run_cell(4)  # backup = clf (aliases the unfitted clf)

    nb_runner.run_cell(5)  # cache HIT -> in-place restore onto the shared object
    assert "RESTORED" in nb_runner.get_output(5), (
        f"fit cell was not a cache hit after restart: {nb_runner.get_output(5)!r}"
    )

    nb_runner.run_cell(6)
    out = nb_runner.get_output(6)
    assert "clf_fit True" in out, out
    # THE assertion: the alias must see the fit. With a rebind it would be False.
    assert "backup_fit True" in out, f"alias did not see the in-place restore: {out!r}"
    assert "same True" in out, f"in-place restore must not rebind the name: {out!r}"


def test_non_estimator_mutation_still_refused(nb_runner):
    """The estimator gate is tight: a bare ``lst.append(x)`` STILL refuses to cache.

    Proves the ``fit``/``partial_fit`` + ``get_params`` gate did not loosen
    general in-place-mutation caching.
    """
    nb_runner.create_notebook([
        SETUP,
        "lst = [1, 2, 3]",
        "lst.append(4)\nprint('len', len(lst))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    out = nb_runner.get_output(3)
    assert "NOT CACHED" in out and "In-place mutation" in out, out

    nb_runner.run_cell(3)  # still refused -- never a cache hit
    assert "NOT CACHED" in nb_runner.get_output(3)


def test_downstream_invalidates_when_data_changes(nb_runner):
    """Editing the upstream data cell invalidates the cached fit + downstream.

    Proves the estimator-fit routing STILL records ``mutation_verdicts`` (which
    the upstream simulation reads to learn that the fit cell produces ``clf``),
    so a change to ``X`` bumps ``clf``'s lineage and the downstream reflects the
    new data instead of replaying the stale cached importances.
    """
    nb_runner.create_notebook([
        SETUP,
        "X, y = make_classification(n_samples=6000, n_features=20, random_state=0)",  # 2
        MODEL,                                                                          # 3
        "clf.fit(X, y)",                                                                # 4
        "print('imp0', round(float(clf.feature_importances_[0]), 6))",                  # 5
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_output(5)
    assert "imp0" in first, first

    # Rebuild X, y from a DIFFERENT random_state (a genuine upstream data change).
    nb_runner.set_cell_source(
        2, "X, y = make_classification(n_samples=6000, n_features=20, random_state=7)"
    )
    nb_runner.run_cell(5)  # re-running downstream must re-fit on the new data
    second = nb_runner.get_output(5)
    assert "imp0" in second, second
    assert first != second, (
        "downstream did not reflect the new data -- the fit was not invalidated "
        f"(mutation_verdicts recording broke?): {first!r} == {second!r}"
    )


def test_bare_fit_pandas_input_restores_repeatedly(nb_runner):
    """CAS-165: a bare fit on a PANDAS DataFrame input is a CLEAN cache hit on
    every warm re-run -- no upstream churn.

    CAS-138's self-referential key (the receiver is both input and output) drifts
    after the fit bumps the receiver's lineage L0->L1. The only thing that recovers
    the pre-fit key today is the upstream checker re-deriving the receiver: it marks
    the receiver a "read-only" mismatch and RE-EXECUTES the constructor upstream on
    *every* warm re-run (cell badge ``EXECUTED`` + an ``Upstream:`` section). That
    incidental cascade is the CAS-165/166 waste ("re-serialises... for nothing").
    The fix resets the receiver's lineage to the virtual (constructor) lineage
    directly, so the re-run is a pure cache HIT with no upstream re-execution.

    Fails without the fix (the ``Upstream:`` constructor re-execution is present);
    passes with it (clean ``RESTORED`` hit, no upstream churn).
    """
    pytest.importorskip("pandas")
    nb_runner.create_notebook([
        SETUP,
        "import pandas as pd\n"
        "rs = np.random.RandomState(0)\n"
        "X_train = pd.DataFrame(rs.rand(4000, 20), columns=[f'f{i}' for i in range(20)])\n"
        "y_train = (X_train['f0'] > 0.5).astype(int)",
        "clf = RandomForestClassifier(n_estimators=160, random_state=0)",
        "clf.fit(X_train, y_train)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    for i in range(3):
        nb_runner.run_cell(4)  # isolated warm re-run -- must be a clean cache hit
        out = nb_runner.get_output(4)
        assert "RESTORED" in out, (
            f"pandas bare fit warm re-run #{i + 1} did not restore: {out!r}"
        )
        # THE discriminator: no upstream constructor re-execution. Without the fix
        # the drifted self-referential key forces the constructor to re-run upstream
        # every warm re-run; with it the receiver's lineage is reset in place.
        assert "Upstream:" not in out, (
            f"pandas bare fit warm re-run #{i + 1} re-executed the constructor "
            f"upstream instead of a clean cache hit (self-referential key drifted): {out!r}"
        )


def test_constructor_edit_invalidates_cached_fit(nb_runner):
    """THE WRONG-result guard: editing the constructor MUST re-run the cached fit.

    The warm-re-run cache hit resets the receiver's lineage to the VIRTUAL
    (constructor-derived) lineage. A constructor edit changes that virtual
    lineage, so the fit's key changes and the fit re-runs on the freshly rebuilt
    estimator -- the downstream reader must reflect the NEW ``n_estimators``, not
    the stale cached value. The check cell is ``@cash:no-cache`` so it reads the
    live namespace instead of replaying its run_all output.
    """
    nb_runner.create_notebook([
        SETUP,
        DATA,
        "clf = RandomForestClassifier(n_estimators=50, random_state=0)",   # 3
        "clf.fit(X, y)",                                                   # 4
        "# @cash:no-cache\nprint('n', clf.n_estimators)",                  # 5
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "n 50" in nb_runner.get_output(5), nb_runner.get_output(5)

    # Edit the constructor 50 -> 200. The cached fit must be invalidated and re-run
    # on the rebuilt estimator (NOT served stale from the n=50 cache entry).
    nb_runner.set_cell_source(
        3, "clf = RandomForestClassifier(n_estimators=200, random_state=0)"
    )
    nb_runner.run_cells([4, 5])
    out = nb_runner.get_output(5)
    assert "n 200" in out, (
        f"constructor edit did not invalidate the cached fit -- stale result served: {out!r}"
    )


def test_partial_fit_stays_cumulative(nb_runner):
    """``partial_fit`` is CUMULATIVE, so it must stay on the value-safe path.

    The fit cheap-reset routing is scoped to ``fit`` only. A lineage-only reset of
    a ``partial_fit`` receiver would either reset or double-count the accumulated
    state on a miss. Two batches (100 + 60 samples) give ``clf.t_ == 161``
    (samples seen + 1); an isolated re-run of the second ``partial_fit`` must keep
    that -- not 61 (reset) nor 221 (double-count). The check cell is
    ``@cash:no-cache`` so it reads the live estimator.
    """
    pytest.importorskip("sklearn")
    nb_runner.create_notebook([
        "import numpy as np\n"
        "from sklearn.linear_model import SGDClassifier\n"
        "import cash\n"
        "%cash_on\n"
        "%cash_badge print",
        "rs = np.random.RandomState(0)\n"
        "Xa = rs.rand(100, 5); ya = (Xa[:, 0] > 0.5).astype(int)\n"
        "Xb = rs.rand(60, 5); yb = (Xb[:, 0] > 0.5).astype(int)",
        "clf = SGDClassifier(random_state=0)",
        "clf.partial_fit(Xa, ya, classes=np.array([0, 1]))",   # 4
        "clf.partial_fit(Xb, yb)",                             # 5
        "# @cash:no-cache\nprint('t', int(clf.t_))",           # 6
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "t 161" in nb_runner.get_output(6), (
        f"cumulative partial_fit setup wrong: {nb_runner.get_output(6)!r}"
    )

    # Isolated re-run of the second partial_fit must not corrupt the cumulative state.
    nb_runner.run_cell(5)
    nb_runner.run_cell(6)
    out = nb_runner.get_output(6)
    assert "t 161" in out, (
        f"partial_fit cumulative state corrupted on isolated re-run "
        f"(expected t 161; 61=reset, 221=double-count): {out!r}"
    )


def test_bare_fit_large_numpy_restores_after_restart(nb_runner):
    """CAS-166: a >8 MiB numpy bare fit RESTORES from disk after a real restart.

    Inputs over ~8 MiB take ``compute_hash``'s sampling path, and the drifting
    self-referential key poisoned the disk entry so it never restored across a
    restart. Forced to disk with ``# @cash:persist`` and run twice pre-restart to
    reproduce the drifted-disk case; after a real kernel restart the rebuilt
    estimator keys off the stable virtual lineage and RESTORES.
    """
    nb_runner.create_notebook([
        SETUP,
        "X = np.random.RandomState(0).rand(110000, 10)\n"   # ~8.8 MiB > 8 MiB
        "y = (X[:, 0] > 0.5).astype(int)",
        "clf = RandomForestClassifier(n_estimators=12, random_state=0)",
        "# @cash:persist\nclf.fit(X, y)\nprint('fitted', clf.n_estimators)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "fitted 12" in nb_runner.get_output(4)
    # Two warm pre-restart runs reproduce the drifted-disk-entry case.
    nb_runner.run_cell(4)
    nb_runner.run_cell(4)

    _restart(nb_runner)
    nb_runner.run_cell(1)  # imports + %cash_on
    nb_runner.run_cell(2)  # X, y (re-derives the same lineage)
    nb_runner.run_cell(3)  # clf (fresh, unfitted)

    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert "RESTORED" in out, (
        f"large bare fit did not restore from disk after restart (CAS-166): {out!r}"
    )
    assert "fitted 12" in out

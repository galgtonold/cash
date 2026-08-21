"""CAS-170 / CAS-138 / CAS-137: how cash treats sklearn's estimator.fit().

Ground-truth probes for the design decision:
  * bare ``model.fit(X, y)``       -> NOT cached by default; re-executes (CAS-170)
  * ``# @cash:cache-fit`` + bare fit -> caches + restores in place (CAS-138, opt-in)
  * ``model = model.fit(X, y)``    -> caches (pure reassignment shape)
  * ``m = Model().fit(X, y)``      -> caches; restores across a real restart

A bare in-place fit re-executes, so the real ``.fit()`` mutates the shared object
rather than a restored copy. Skipping the cache is also net-NEUTRAL rather than
net-negative: the model is never serialised, so a perpetually-missing fit cannot
cost more than it saves.

**Withdrawn (CAS-184):** this file used to claim that a re-executing fit made
aliases "correct BY CONSTRUCTION". It does not. ``backup = clf`` is an ORDINARY
assignment, so cash cached *that statement* and restored ``backup`` to a
deserialised copy taken before the fit -- the alias broke at a statement the fit
has no bearing on. The fit was innocent. CAS-184 fixes it by refusing to cache a
bare alias bind at all; ``test_bare_fit_alias_survives_warm_reruns`` below is the
guard, and it only fails from the SECOND warm re-run, which is how the original
one-repetition test came to confirm the wrong belief.

``# @cash:cache-fit`` opts back in to the CAS-138 machinery for users who want it.
"""
import pytest
from conftest import shows_cached

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
    """Restart the kernel in place and re-inject the notebook path.

    Delegates to the runner rather than driving the KernelManager directly.
    CAS-190 added ``nb_runner.restart()`` precisely because nine files had
    hand-rolled this and each was free to get the after-care wrong; this was
    one of them. A restart puts the kernel back at the cwd its PROCESS was
    launched with, and under CASH_TEST_REUSE_KERNEL=1 that is the repo root,
    not this test's tmp dir -- so cash rebuilds its backend against
    ``<repo>/.cash`` while the entry written before the restart sits in
    ``<tmp>/.cash``, and nothing restores. Measured: the three
    restores-after-restart tests here failed with "COMPUTED ... -> RAM+DISK"
    where they expected RESTORED.
    """
    nb_runner.restart()


# ----------------------------------------------------------------------
# The DEFAULT: a bare fit is not cached
# ----------------------------------------------------------------------

def test_bare_fit_not_cached_by_default(nb_runner):
    """CAS-170: a bare ``clf.fit(X, y)`` is NOT CACHED and re-executes every run.

    The inverse of the old ``test_bare_fit_now_caches``. CAS-138 routed a bare fit
    to caching by default; four rounds of user testing found the correctness
    surface exceeds what per-statement restore can guarantee (stale aliases) and
    that the canonical paths perpetually MISS -- re-serialising the model every run
    for a NET LOSS. So the default reverts to the general in-place-mutation path:
    skip-cached, receiver lineage bumped, statement re-executes.

    The badge reason IS the net-neutrality proof: ``skip_cache`` is what gates
    ``_save_to_cache``, so a refused statement never serialises the model at all.

    The fit cell holds exactly ONE statement so a bare ``not shows_cached(...)`` check
    is unambiguous -- a second statement in the same cell (e.g. a ``print``) caches
    on its own and would put an unrelated ``CACHED`` line in the same badge.
    """
    nb_runner.create_notebook([
        SETUP, DATA, MODEL,
        "clf.fit(X, y)",                                    # 4 (single statement)
        "# @cash:no-cache\nprint('fitted', clf.n_estimators)",   # 5
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    cold = nb_runner.get_output(4)
    assert "NOT CACHED" in cold, f"bare fit should not cache by default: {cold!r}"
    assert "In-place mutation" in cold, cold
    assert "fitted 160" in nb_runner.get_output(5)

    nb_runner.run_cell(4)  # isolated re-run must RE-EXECUTE, never restore
    warm = nb_runner.get_output(4)
    assert not shows_cached(warm), (
        f"bare fit restored from cache without @cash:cache-fit: {warm!r}"
    )
    assert "NOT CACHED" in warm, warm
    assert "In-place mutation" in warm, warm

    nb_runner.run_cell(5)
    assert "fitted 160" in nb_runner.get_output(5)


def test_bare_fit_alias_survives_warm_reruns(nb_runner):
    """THE WRONG-result guard, in the shape a user-tester actually hit (CAS-184).

    ``backup = clf`` aliases the estimator, then the bare fit mutates the shared
    object. A plain kernel gives ``backup is clf`` -> True and a fitted backup on
    every run. Cash must agree.

    This is the REAL-WORLD half of the CAS-184 guard (the mechanism is pinned
    deterministically in ``test_alias_assignment_identity.py``): a fitted
    RandomForest is slow enough to hash that the alias bind's measured time clears
    the 10ms ``min_execution_time_to_cache_seconds`` floor, so cash writes a cache
    entry for it WITHOUT any ``@cash:persist`` -- which is why this reproduced in
    the wild on an ordinary notebook.

    Two things this test's ancestor got wrong, both load-bearing:

    * It ran ONE warm re-run of the FIT cell and never re-ran the alias cell. The
      alias restore lands from the SECOND warm re-run, so it passed and was read
      as proof that aliases were "correct by construction". They are not; the fit
      was innocent and the assignment was the bug. Hence ``run_all`` x4.
    * Its probe cell put ONE ``@cash:no-cache`` above THREE prints. The directive
      binds to a single statement (it walks back over consecutive COMMENT lines
      only), so prints 2 and 3 cached and replayed -- the cell printed a stale
      ``backup_fit True`` next to a live ``same False``. Every print now carries
      its own directive, and the test asserts nothing in the probe was CACHED.
    """
    nb_runner.create_notebook([
        SETUP, DATA, MODEL,                       # 1, 2, 3
        "backup = clf",                           # 4
        "clf.fit(X, y)",                          # 5
        "# @cash:no-cache\n"                      # 6
        "print('clf_fit', hasattr(clf, 'classes_'))\n"
        "# @cash:no-cache\n"
        "print('backup_fit', hasattr(backup, 'classes_'))\n"
        "# @cash:no-cache\n"
        "print('same', clf is backup)",
    ])
    nb_runner.start_kernel()

    for rep in range(4):  # 1 cold + 3 warm; the alias restore lands on warm #2
        nb_runner.run_all()

        assert not shows_cached(nb_runner.get_output(5)), (
            f"rep {rep}: bare fit must not restore by default: "
            f"{nb_runner.get_output(5)!r}"
        )
        out = nb_runner.get_output(6)
        assert not shows_cached(out), (
            f"rep {rep}: the probe cell replayed from cache instead of reading the "
            f"live namespace -- this probe is invalid, not passing: {out!r}"
        )
        assert "clf_fit True" in out, f"rep {rep}: {out!r}"
        # THE assertions: identity is preserved and the alias is fitted.
        assert "same True" in out, (
            f"rep {rep}: `backup = clf` restored a copy, so the alias is no longer "
            f"the fitted estimator (CAS-184): {out!r}"
        )
        assert "backup_fit True" in out, (
            f"rep {rep}: alias left stale/unfitted (CAS-184): {out!r}"
        )


# ----------------------------------------------------------------------
# The OPT-IN: # @cash:cache-fit restores the CAS-138 behaviour
# ----------------------------------------------------------------------

def test_cache_fit_annotation_opts_in(nb_runner):
    """``# @cash:cache-fit`` turns the CAS-138 path back on for one statement.

    The twin of ``test_bare_fit_not_cached_by_default``: the same notebook with
    the directive added must land a cache HIT on an isolated re-run. The receiver
    is self-referential (input + output), so the re-run re-derives the unfitted
    estimator upstream and the fit cell restores.
    """
    nb_runner.create_notebook([
        SETUP, DATA, MODEL,
        "# @cash:cache-fit\nclf.fit(X, y)\nprint('fitted', clf.n_estimators)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.run_cell(4)  # isolated re-run should be a cache hit
    out = nb_runner.get_output(4)
    assert shows_cached(out), f"@cash:cache-fit did not cache the bare fit: {out!r}"
    assert "fitted 160" in out


def test_cache_fit_restores_in_place_when_receiver_is_live(nb_runner):
    """CAS-138 GATE (opt-in): the ``cache-fit`` hit transfers state onto the
    EXISTING receiver rather than rebinding, so a live alias sees the fit.

    This is the in-place restore doing its job in the case it CAN handle: the
    receiver present in ``user_ns`` at restore time is the one ``backup`` aliases,
    so ``__setstate__`` onto it is observed through both names.

    It is NOT a general guarantee, and the opt-in is documented as such: the
    restore is per statement, so if the CONSTRUCTOR statement's own cache hit
    rebinds the receiver first, the alias graph is already broken upstream and no
    in-place transfer here can repair it. That is exactly why this path is opt-in
    and the default (``test_bare_fit_alias_survives_warm_reruns``) is not.

    Deterministic setup via a real restart: cells 1-4 run in order so ``backup``
    aliases the freshly rebuilt (unfitted) ``clf`` right before the fit cell,
    which then lands a persisted-cache HIT.
    """
    nb_runner.create_notebook([
        SETUP, DATA, MODEL,                       # 1, 2, 3
        "backup = clf",                           # 4
        "# @cash:cache-fit\n"                     # 5
        "# @cash:persist\n"
        "clf.fit(X, y)",
        "# @cash:no-cache\n"                      # 6
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
    assert shows_cached(nb_runner.get_output(5)), (
        f"fit cell was not a cache hit after restart: {nb_runner.get_output(5)!r}"
    )

    nb_runner.run_cell(6)
    out = nb_runner.get_output(6)
    assert "clf_fit True" in out, out
    # THE assertion: the alias must see the fit. With a rebind it would be False.
    assert "backup_fit True" in out, f"alias did not see the in-place restore: {out!r}"
    assert "same True" in out, f"in-place restore must not rebind the name: {out!r}"


def test_cache_fit_pandas_input_restores_repeatedly(nb_runner):
    """CAS-165 (opt-in): a ``cache-fit`` bare fit on a PANDAS DataFrame input is a
    CLEAN cache hit on every warm re-run -- no upstream churn.

    CAS-138's self-referential key (the receiver is both input and output) drifts
    after the fit bumps the receiver's lineage L0->L1. The only thing that recovers
    the pre-fit key otherwise is the upstream checker re-deriving the receiver: it
    marks the receiver a "read-only" mismatch and RE-EXECUTES the constructor
    upstream on *every* warm re-run (cell badge ``EXECUTED`` + an ``Upstream:``
    section). That incidental cascade is the CAS-165/166 waste. The fix resets the
    receiver's lineage to the virtual (constructor) lineage directly, so the re-run
    is a pure cache HIT with no upstream re-execution.

    Still load-bearing for the opt-in path: without it, opting in is a net LOSS.
    """
    pytest.importorskip("pandas")
    nb_runner.create_notebook([
        SETUP,
        "import pandas as pd\n"
        "rs = np.random.RandomState(0)\n"
        "X_train = pd.DataFrame(rs.rand(4000, 20), columns=[f'f{i}' for i in range(20)])\n"
        "y_train = (X_train['f0'] > 0.5).astype(int)",
        "clf = RandomForestClassifier(n_estimators=160, random_state=0)",
        "# @cash:cache-fit\nclf.fit(X_train, y_train)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    for i in range(3):
        nb_runner.run_cell(4)  # isolated warm re-run -- must be a clean cache hit
        out = nb_runner.get_output(4)
        assert shows_cached(out), (
            f"pandas cache-fit warm re-run #{i + 1} did not restore: {out!r}"
        )
        # THE discriminator: no upstream constructor re-execution. Without the fix
        # the drifted self-referential key forces the constructor to re-run upstream
        # every warm re-run; with it the receiver's lineage is reset in place.
        assert "Upstream:" not in out, (
            f"pandas cache-fit warm re-run #{i + 1} re-executed the constructor "
            f"upstream instead of a clean cache hit (self-referential key drifted): {out!r}"
        )


def test_cache_fit_make_classification_split_restores_repeatedly(nb_runner):
    """CAS-171: the canonical beginner ML chain warm-re-runs as a CLEAN cache hit.

    ``make_classification`` -> ``pd.DataFrame`` -> ``train_test_split`` -> bare
    ``cache-fit``: the shape a user-tester reported as a perpetual MISS, blamed on
    ``make_classification`` specifically (an rng-derived control cached, so the
    data SOURCE looked like the discriminator).

    Measurement says otherwise: the source is irrelevant. Swapping
    ``make_classification`` for ``rng.standard_normal`` or a local function
    produces byte-identical caching behaviour in every configuration. What this
    chain actually exercises is CAS-165/166's self-referential fit key, and the
    discriminator is the ``Upstream:`` cascade, not the callable: with the
    fit-receiver lineage reset disabled, ALL sources re-derive the whole upstream
    chain on every warm re-run; with it, all of them land a clean hit.

    So this is not a ``make_classification`` regression test -- it is the
    CAS-165/166 guard extended along the axis the tester's report actually
    covered and ``test_cache_fit_pandas_input_restores_repeatedly`` does not: a
    multi-output ``train_test_split`` between the frame and the fit, whose four
    unpacked outputs each carry lineage into the self-referential key.
    """
    pytest.importorskip("pandas")
    nb_runner.create_notebook([
        SETUP,
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split",
        # >8 MiB so the fit inputs take compute_hash's sampling path (CAS-166).
        "X, y = make_classification(n_samples=60000, n_features=20, random_state=42)",
        "df = pd.DataFrame(X, columns=[f'f{i}' for i in range(20)])",
        "X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42)",
        "clf = RandomForestClassifier(n_estimators=40, random_state=42)",
        "# @cash:cache-fit\n# @cash:persist\nclf.fit(X_train, y_train)",   # 7
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    for i in range(3):
        nb_runner.run_cell(7)  # isolated warm re-run -- must be a clean cache hit
        out = nb_runner.get_output(7)
        assert shows_cached(out), (
            f"make_classification cache-fit warm re-run #{i + 1} did not restore "
            f"(CAS-171): {out!r}"
        )
        # THE discriminator, and the half a plain shows_cached() check would miss: a
        # full upstream re-derivation restores too, so without this a silent
        # re-run of the whole make_classification -> split chain every warm
        # re-run would pass as a technically-correct hit.
        assert "Upstream:" not in out, (
            f"make_classification cache-fit warm re-run #{i + 1} re-executed the "
            f"upstream chain instead of a clean cache hit: {out!r}"
        )


def test_cache_fit_constructor_edit_invalidates_cached_fit(nb_runner):
    """THE WRONG-result guard for the opt-in: editing the constructor MUST re-run
    the cached fit.

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
        "# @cash:cache-fit\nclf.fit(X, y)",                                # 4
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


def test_cache_fit_large_numpy_restores_after_restart(nb_runner):
    """CAS-166 (opt-in): a >8 MiB numpy ``cache-fit`` bare fit RESTORES from disk
    after a real restart.

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
        "# @cash:cache-fit\n# @cash:persist\nclf.fit(X, y)\nprint('fitted', clf.n_estimators)",
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
    assert shows_cached(out), (
        f"large cache-fit did not restore from disk after restart (CAS-166): {out!r}"
    )
    assert "fitted 12" in out


# ----------------------------------------------------------------------
# Shapes that cache regardless of the directive (normal assignments)
# ----------------------------------------------------------------------

def test_reassign_fit_caches(nb_runner):
    """Form 2: ``clf = clf.fit(X, y)`` (add ``clf = ``) caches and restores on re-run.

    An ordinary ``Assign`` -- no bare-Expr receiver, so the estimator-fit gate and
    the in-place-mutation classifier are both irrelevant. It must keep caching
    WITHOUT any directive: the CAS-170 demotion is scoped to the bare-Expr form.
    """
    nb_runner.create_notebook([
        SETUP, DATA, MODEL,
        "clf = clf.fit(X, y)\nprint('fitted', clf.n_estimators)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.run_cell(4)  # isolated re-run should be a cache hit
    out = nb_runner.get_output(4)
    assert shows_cached(out), f"reassign-fit did not cache: {out!r}"


def test_construct_fit_assign_restores_after_restart(nb_runner):
    """Form 3 + CAS-137: ``m = Model().fit(X, y)`` restores after a REAL restart.

    Also an ordinary ``Assign`` (the receiver is a temporary, never a name), so it
    caches with no directive. Forced to disk with ``# @cash:persist`` so the test
    probes the restore path itself, not the cost model's RAM-only decision for a
    cheap fit.
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
    assert shows_cached(out), f"fitted estimator did not restore after restart (CAS-137): {out!r}"
    assert "n = 160" in out


# ----------------------------------------------------------------------
# Gate tightness + lineage plumbing (unchanged by CAS-170)
# ----------------------------------------------------------------------

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
    """Editing the upstream data cell invalidates the fit's downstream consumer.

    Proves the bare fit STILL records ``mutation_verdicts`` (which the upstream
    simulation reads to learn that the fit cell produces ``clf``), so a change to
    ``X`` bumps ``clf``'s lineage and the downstream reflects the new data instead
    of replaying the stale cached importances. Holds on the default (skip-cached)
    path: the lineage bump is what carries the change, not the cache entry.
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

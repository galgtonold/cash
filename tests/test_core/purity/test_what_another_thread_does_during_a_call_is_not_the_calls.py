"""What another thread does while a cached call runs is not blamed on the
function.

The environment, a logger's level, the global random streams: the process
keeps one of each, so comparing them around a body also sees what a
heartbeat or server thread did meanwhile. One such sight made the function
run uncached for the rest of the process ("the call changed the
environment"), or marked it, for good and across processes, as drawing
random numbers it never drew: a miss at every seed anyone set, and
RANDOM-UNSEEDED about a function that draws nothing.

Here the other thread acts exactly while the body runs (the body waits for
it), and the call's own doing is still found: by its thread where the
change is a counted call, on the next computed call where it is not.
"""

from __future__ import annotations

import atexit
import logging
import os
import random
import threading
import time
import warnings

import pytest

from cash import Cash
from cash.exceptions import CashCacheStoreFailedWarning, CashImpurityWarning
from cash.tracking import process_registries
from cash.tracking.randomness import CashRandomnessWarning, lineage

_KEY = "ZZ_SET_BESIDE_A_CACHED_CALL"
_LOGGER = "zz.set_beside_a_cached_call"


@pytest.fixture(autouse=True)
def as_found(monkeypatch):
    """No seed left by another test, and each thing put back as it was."""
    monkeypatch.setattr(lineage, "_ACTIVE_SEED_EPOCHS", None)
    monkeypatch.setattr(lineage, "_PROCESS_SEEDED", set())
    monkeypatch.setattr(lineage, "_TORCH_SEEDED", set())
    state = random.getstate()
    yield
    random.setstate(state)
    os.environ.pop(_KEY, None)
    logging.getLogger(_LOGGER).setLevel(0)


@pytest.fixture
def inst(tmp_path):
    return _cash(tmp_path)


def _cash(tmp_path):
    made = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    # What a body captures to let the other thread act: the same to every key.
    made.register_hasher(_Beside, lambda beside: "the thread beside")
    return made


class _Beside:
    """A thread that does *act* each time a body lets it (`during`), and
    nothing in between."""

    def __init__(self, act):
        self._act = act
        self._go = threading.Event()
        self._done = threading.Event()
        self._over = False
        self._thread = threading.Thread(target=self._run, name="zz-beside")
        self._thread.start()

    def _run(self):
        while True:
            self._go.wait()
            self._go.clear()
            if self._over:
                return
            self._act()
            self._done.set()

    def during(self):
        """Called by a body: the thread acts now, and has acted on return."""
        self._done.clear()
        self._go.set()
        if not self._done.wait(20):
            raise RuntimeError("the other thread did not act")

    def end(self):
        self._over = True
        self._go.set()
        self._thread.join(20)


def _codes(record, kind):
    return [str(w.message).split("]")[0].strip("[") for w in record if issubclass(type(w.message), kind)]


def _set_environment():
    os.environ[_KEY] = str(random.Random().random())


_callbacks = []


def _register_at_exit():
    def never_run():
        return None

    _callbacks.append(never_run)
    atexit.register(never_run)


@pytest.mark.parametrize("act", [_set_environment, lambda: os.putenv(_KEY, "1"), _register_at_exit])
def test_a_counted_change_another_thread_makes_leaves_the_call_stored(inst, act):
    beside = _Beside(act)
    ran = []

    @inst.cache
    def doubled(x):
        ran.append(x)
        beside.during()
        return x * 2

    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            assert doubled(1) == 2
            assert doubled(1) == 2
            assert doubled(5) == 10
            assert doubled(5) == 10
    finally:
        beside.end()
        while _callbacks:
            atexit.unregister(_callbacks.pop())
    assert ran == [1, 5], "the calls another thread ran beside were not stored"
    assert "IMPURE-OBSERVED-EFFECTS" not in _codes(record, CashImpurityWarning)
    assert _codes(record, CashCacheStoreFailedWarning) == []


def test_the_calls_own_write_beside_another_thread_is_still_the_calls(inst):
    beside = _Beside(lambda: None)
    ran = []

    @inst.cache
    def sets(x):
        ran.append(x)
        beside.during()
        os.environ[_KEY] = "set"
        return x * 2

    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            for _ in range(3):
                os.environ.pop(_KEY, None)
                assert sets(1) == 2
                assert os.environ.get(_KEY) == "set", "served without making the setting"
    finally:
        beside.end()
    assert ran == [1, 1, 1]
    assert "IMPURE-OBSERVED-EFFECTS" in _codes(record, CashImpurityWarning)


def test_a_write_by_a_thread_the_call_started_is_the_calls(inst):
    beside = _Beside(lambda: None)
    ran = []

    @inst.cache
    def sets_on_a_thread(x):
        ran.append(x)
        worker = threading.Thread(target=_set_environment)
        worker.start()
        worker.join()
        return x * 2

    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            for _ in range(3):
                os.environ.pop(_KEY, None)
                assert sets_on_a_thread(1) == 2
                assert _KEY in os.environ, "served without making the setting"
    finally:
        beside.end()
    assert ran == [1, 1, 1]
    assert "IMPURE-OBSERVED-EFFECTS" in _codes(record, CashImpurityWarning)


def test_a_change_that_cannot_be_told_apart_leaves_one_call_unstored(inst):
    levels = iter(range(11, 99))
    acts = [True]
    beside = _Beside(lambda: acts[0] and logging.getLogger(_LOGGER).setLevel(next(levels)))
    ran = []

    @inst.cache
    def doubled(x):
        ran.append(x)
        beside.during()
        return x * 2

    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            assert doubled(1) == 2
            # Which of them set the level cannot be told: not stored, said once.
            assert _codes(record, CashCacheStoreFailedWarning) == ["STORE-STATE-CHANGED"]
            acts[0] = False
            assert doubled(1) == 2
            assert doubled(1) == 2
            assert doubled(5) == 10
            assert doubled(5) == 10
    finally:
        beside.end()
    assert ran == [1, 1, 5], "the function went on running uncached"
    assert "IMPURE-OBSERVED-EFFECTS" not in _codes(record, CashImpurityWarning)
    assert _codes(record, CashCacheStoreFailedWarning) == ["STORE-STATE-CHANGED"]


def test_the_functions_own_change_is_found_on_its_next_call(inst):
    beside = _Beside(lambda: None)
    ran = []

    @inst.cache
    def sets_a_level(x):
        ran.append(x)
        beside.during()
        logging.getLogger(_LOGGER).setLevel(10)
        return x * 2

    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            for _ in range(4):
                logging.getLogger(_LOGGER).setLevel(0)
                assert sets_a_level(1) == 2
                assert logging.getLogger(_LOGGER).level == 10, "served without making the setting"
    finally:
        beside.end()
    assert ran == [1, 1, 1, 1]
    said = [str(w.message) for w in record if issubclass(type(w.message), CashImpurityWarning)]
    assert any("IMPURE-OBSERVED-EFFECTS" in text and "a logger's level" in text for text in said), said


def test_a_change_made_with_no_other_thread_is_the_calls_at_once(inst, monkeypatch):
    # Alone whatever else the test run left alive in this process.
    monkeypatch.setattr(process_registries, "_working_threads", list)
    monkeypatch.setattr(process_registries, "_others_run", lambda: False)

    @inst.cache
    def sets_a_level(x):
        logging.getLogger(_LOGGER).setLevel(10)
        return x * 2

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert sets_a_level(1) == 2
    assert "IMPURE-OBSERVED-EFFECTS" in _codes(record, CashImpurityWarning)
    assert _codes(record, CashCacheStoreFailedWarning) == []


@pytest.mark.parametrize("seeded", [True, False], ids=["seeded", "unseeded"])
def test_numbers_another_thread_draws_do_not_mark_the_function_as_drawing(inst, seeded):
    if seeded:
        random.seed(1)
    acts = [True]
    beside = _Beside(lambda: acts[0] and random.random())
    ran = []

    @inst.cache
    def doubled(x):
        ran.append(x)
        beside.during()
        return x * 2

    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            assert doubled(1) == 2  # may have drawn, for all the stream shows: not stored
            acts[0] = False
            assert doubled(1) == 2
            assert doubled(1) == 2
            if seeded:
                # Not keyed by a stream it never drew from.
                random.seed(2)
            random.random()
            assert doubled(1) == 2
            assert doubled(5) == 10
            assert doubled(5) == 10
    finally:
        beside.end()
    assert ran == [1, 1, 5]
    assert _codes(record, CashRandomnessWarning) == []
    assert "code or state changed" not in str(doubled.cache_info()["miss_reasons"])


def test_a_function_that_draws_beside_another_thread_is_found_on_its_next_call(inst):
    beside = _Beside(lambda: None)
    ran = []

    @inst.cache
    def helper_draws(x):
        ran.append(x)
        beside.during()
        return x + _draw()

    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            first = helper_draws(1)
            assert _codes(record, CashRandomnessWarning) == []
            second = helper_draws(1)
            assert second != first, "a draw that might be the call's was stored unsaid"
            assert "RANDOM-UNSEEDED" in _codes(record, CashRandomnessWarning)
            # Said, and frozen from here on as an unseeded draw is.
            assert helper_draws(1) == second
    finally:
        beside.end()
    assert ran == [1, 1]


def test_an_estimator_left_without_random_state_is_frozen_from_its_first_fit_beside_another_thread(inst):
    """The source already says the fit draws: RANDOM-UNSEEDED at decoration
    names ``RandomForestClassifier()`` and promises the first call is frozen.
    A thread working beside that first fit leaves nothing to tell apart, so
    it is stored; refused, the second fit (a different draw) was frozen
    instead, and a test run with a kernel's heartbeat thread still alive
    saw two different feature importances."""
    ensemble = pytest.importorskip("sklearn.ensemble")
    import numpy as np

    beside = _Beside(lambda: None)
    X = np.random.RandomState(0).rand(60, 4)
    y = (X[:, 0] > 0.5).astype(int)
    ran = []

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")

        @inst.cache
        def fit_model(n):
            ran.append(n)
            beside.during()
            m = ensemble.RandomForestClassifier(n_estimators=n)  # no random_state
            m.fit(X, y)
            return m.feature_importances_.tolist()

    try:
        assert "RANDOM-UNSEEDED" in _codes(record, CashRandomnessWarning)
        assert fit_model(5) == fit_model(5), "the first fit was not the one frozen"
    finally:
        beside.end()
    assert ran == [5]


def test_the_open_question_is_settled_by_the_next_process(tmp_path):
    """A function called once a process: the second run's draw settles it."""
    beside = _Beside(lambda: None)

    def helper_draws(x):
        beside.during()
        return x + _draw()

    try:
        seen = []
        for _run in range(3):
            inst = _cash(tmp_path)
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                seen.append(inst.cache(helper_draws)(1))
            if _run == 0:
                assert _codes(record, CashRandomnessWarning) == []
            if _run == 1:
                assert "RANDOM-UNSEEDED" in _codes(record, CashRandomnessWarning)
    finally:
        beside.end()
    assert seen[0] != seen[1], "a draw that might be the call's was stored unsaid"
    assert seen[2] == seen[1]


def _draw():
    return _pick()


def _pick():
    return random.random()


class _Asleep:
    """A thread of the user's that is alive and does nothing: a pool's
    idle worker, a monitor between two looks."""

    def __init__(self):
        self._wake = threading.Event()
        self._waiting = threading.Event()
        self._thread = threading.Thread(target=self._run, name="zz-asleep")
        self._thread.start()
        assert self._waiting.wait(20)

    def _run(self):
        self._waiting.set()
        self._wake.wait()

    def end(self):
        self._wake.set()
        self._thread.join(20)


def _settled(asleep):
    """Once *asleep*'s thread uses no more processor time: it is blocked."""
    clock = process_registries._cpu_clock(asleep._thread.native_id)
    last = -1
    for _ in range(2000):
        used = time.clock_gettime_ns(clock)
        if used == last:
            return
        last = used
        time.sleep(0.005)
    raise AssertionError("the sleeping thread kept running")


@pytest.fixture
def only_this_tests_threads(monkeypatch):
    """Whatever else the test run left alive in this process is not asked."""
    real = process_registries._working_threads
    monkeypatch.setattr(
        process_registries, "_working_threads", lambda: [t for t in real() or () if t.name.startswith("zz-")]
    )


proved_idle = pytest.mark.skipif(not process_registries._CPU_CLOCKS, reason="per-thread processor time: Linux")


@proved_idle
def test_a_thread_that_used_no_processor_time_is_not_another_writer(inst, only_this_tests_threads):
    asleep = _Asleep()

    @inst.cache
    def sets_a_level(x):
        logging.getLogger(_LOGGER).setLevel(10)
        return x * 2

    @inst.cache
    def helper_draws(x):
        return x + _draw()

    try:
        _settled(asleep)
        assert process_registries._working_threads(), "the sleeping thread does not count as another: nothing is tested"
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            assert sets_a_level(1) == 2
            helper_draws(1)
    finally:
        asleep.end()
    assert "IMPURE-OBSERVED-EFFECTS" in _codes(record, CashImpurityWarning), "not the call's at once"
    assert _codes(record, CashCacheStoreFailedWarning) == []
    assert "RANDOM-UNSEEDED" in _codes(record, CashRandomnessWarning), "not the call's draw at once"


@proved_idle
def test_the_proof_needs_every_other_thread_idle_from_before_the_body(only_this_tests_threads):
    asleep = _Asleep()
    try:
        _settled(asleep)
        before = process_registries.threads_beside()
        assert before.others and before.clocks
        assert not process_registries.beside_threads(before), "an idle thread was counted as a writer"

        # One that ran, however briefly.
        ran = _Beside(lambda: None)
        try:
            before = process_registries.threads_beside()
            ran.during()
            assert process_registries.beside_threads(before)
        finally:
            ran.end()

        # One started since, and one that ended since.
        before = process_registries.threads_beside()
        late = _Asleep()
        try:
            assert process_registries.beside_threads(before)
            before = process_registries.threads_beside()
        finally:
            late.end()
        assert process_registries.beside_threads(before)
    finally:
        asleep.end()


def test_where_processor_time_cannot_be_read_an_idle_thread_still_counts(monkeypatch, only_this_tests_threads):
    monkeypatch.setattr(process_registries, "_CPU_CLOCKS", False)
    asleep = _Asleep()
    try:
        before = process_registries.threads_beside()
        assert before.others and before.clocks is None
        assert process_registries.beside_threads(before)
    finally:
        asleep.end()

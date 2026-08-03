"""CAS-260: a global written INSIDE a cached callee is captured and restored,
the same way the identical write is when the user spells it inline.

    CALLS = []
    def compute(v):
        CALLS.append(v)      # not in the statement's source -- invisible to
        return v * 10        # every gate that reads the statement's AST

Two mechanisms, split by what each can guarantee:

* **The statement is skip-cached** and re-executes, so the write really
  happens. This is exactly what the INLINE spelling of the same mutation
  already gets (``_classify_method_mutations`` routes an in-place mutation to
  ``mut_pre_route``: bump the lineage, skip the cache, re-execute).
* **The call inside it is still cached**, keyed on the mutated global's own
  pre-call state, and restores that call's post-call state on a hit. So the
  expensive work is served and only the glue re-runs.

An earlier version captured the global as an output of the STATEMENT and
restored it, copying ``force_outputs``. That is unsound the moment a global has
more than one writer -- an absolute end state does not compose with a prefix
that was itself skipped. Measured, two calls to one appending helper in a
single cell::

    expected  ['ok:3.3', 'cleanup', 'err:zero_div', 'cleanup']
    observed  ['err:zero_div', 'cleanup']

``force_outputs`` can restore an accumulator, but only under
``cacheable_accumulator_loop``'s conditions -- fresh empty seed, a single
accumulator call -- which are precisely the guarantees that make one writer's
snapshot sufficient. The per-call route needs no such guarantee because each
call pins its own pre-state.

**Inline is the contract.** Every test here runs the inline spelling beside the
in-callee one, in one notebook and one kernel, and asserts they agree. That is
the actual defect statement -- not "cash caches something impure" but "the same
code behaves differently inline and in a function" -- and it makes the
assertions independent of what cash's re-run semantics happen to be.

Three measurement traps, all of which produced a wrong reading during this
work and all of which are avoided below:

* **A printed value is replayed stdout on a hit**, not a live read. Reading
  ``CALLS`` through ``print(...)`` in a notebook cell reports what was on
  screen the run the entry was written. Every assertion here reads out of band
  via :func:`_peek`, and with ``globals().get(name)`` rather than a bare name.
* **``open(p, 'a')`` as an execution counter registers as a file DEPENDENCY**
  of the entry, so the file changes every run, the entry is never fresh, and
  the call re-runs forever -- the instrument silently disables what it
  measures. The counters here go through ``os.open``/``os.write``, which
  ``FileAccessTracker`` does not patch.
* **Two floors, not one.** ``min_execution_time_to_cache_seconds`` (0.01s) and
  the ~0.1s persistence floor below which an entry stays in RAM and never
  reaches disk. At 30ms the PURE control re-ran after every restart -- a
  durability floor reading as a correctness finding. Hence 150ms bodies.
"""
import pytest

pytestmark = [pytest.mark.integration]

SETUP = "import cash\n%cash_on\n"

# Cheap enough to be a rounding error, dear enough to clear both floors.
_BODY_MS = 150


def _defs(cp, cf, ci, *, mult=10):
    """Three callees, three counters, one shared shape.

    ``compute_i`` is pure and paired with an inline ``CALLS_I.append`` at the
    call site; ``compute_f`` does the identical append INSIDE its body;
    ``compute_p`` is neither and exists only to prove the harness is measuring
    cache behaviour at all.
    """
    return (
        "import time, os\n"
        "def _tick(p):\n"
        "    fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
        "    os.write(fd, b'X')\n"
        "    os.close(fd)\n"
        "def _busy(ms):\n"
        "    t = time.perf_counter() + ms / 1000.0\n"
        "    while time.perf_counter() < t:\n"
        "        pass\n"
        "CALLS_I = []\n"
        "CALLS_F = []\n"
        "def compute_p(v):\n"
        f"    _tick(r'{cp}')\n"
        f"    _busy({_BODY_MS})\n"
        f"    return v * {mult}\n"
        "def compute_f(v):\n"
        "    CALLS_F.append(v)\n"
        f"    _tick(r'{cf}')\n"
        f"    _busy({_BODY_MS})\n"
        f"    return v * {mult}\n"
        "def compute_i(v):\n"
        f"    _tick(r'{ci}')\n"
        f"    _busy({_BODY_MS})\n"
        f"    return v * {mult}\n"
    )


ARM_P = "ap = compute_p(1)\n"
ARM_F = "af = compute_f(1)\n"
ARM_I = "CALLS_I.append(1)\nai = compute_i(1)\n"

DEFS_CELL = 2
ARMS = [3, 4, 5]


def _n(path):
    return len(path.read_bytes()) if path.exists() else 0


def _peek(nb_runner, expr):
    """Evaluate *expr* in the live kernel, outside the notebook's cells.

    ``store_history=False``, so nothing about this execution is a notebook cell
    and nothing about it is cached or replayed. Always call it with
    ``globals().get(name)``: a bare undefined name raises NameError, produces no
    stdout, and the "no output" answer then reads as a value rather than as a
    lookup failure.
    """
    seen = []

    def _hook(msg):
        if msg['msg_type'] == 'stream' and msg['content'].get('name') == 'stdout':
            seen.append(msg['content']['text'])

    nb_runner._run_async(nb_runner.client.kc._async_execute_interactive(
        f"print('__PEEK__', repr({expr}))", store_history=False, output_hook=_hook))
    for line in "".join(seen).splitlines():
        if '__PEEK__' in line:
            return line.split('__PEEK__', 1)[1].strip()
    return "<no output>"


def _globals(nb_runner):
    return {
        "inline": _peek(nb_runner, "globals().get('CALLS_I')"),
        "in_callee": _peek(nb_runner, "globals().get('CALLS_F')"),
    }


def _start(nb_runner, tmp_path, cells=None):
    cp, cf, ci = (tmp_path / f"{n}.log" for n in ("p", "f", "i"))
    nb_runner.create_notebook(cells or [SETUP, _defs(cp, cf, ci), ARM_P, ARM_F, ARM_I])
    nb_runner.start_kernel()
    nb_runner.run_all()
    return cp, cf, ci


def test_a_callee_written_global_survives_a_kernel_restart(nb_runner, tmp_path):
    """The core defect. After a restart the seed re-runs, so the global starts
    empty and the statement hits -- which is exactly when a write that is not an
    output disappears.

    Measured before this landed::

        P (pure control)  hits, correct
        I (inline)        hits, CALLS_I == [1]     restored
        F (in-callee)     hits, CALLS_F == []      dropped
    """
    cp, cf, ci = _start(nb_runner, tmp_path)
    assert _globals(nb_runner) == {"inline": "[1]", "in_callee": "[1]"}, "cold run is wrong"
    cold = [_n(cp), _n(cf), _n(ci)]
    assert cold == [1, 1, 1], f"cold run did not execute each callee once: {cold}"

    nb_runner.restart()
    nb_runner.run_all()

    served = [a - b for a, b in zip([_n(cp), _n(cf), _n(ci)], cold)]
    assert served == [0, 0, 0], (
        f"nothing was served from cache after the restart ({served} real calls); "
        "the assertion below would pass vacuously. Check the ~0.1s persistence "
        "floor before reading anything into it."
    )
    assert _globals(nb_runner) == {"inline": "[1]", "in_callee": "[1]"}, (
        "the callee's write to CALLS_F did not survive a restart, while the "
        "identical write spelled inline did"
    )


def test_a_same_session_rerun_neither_freezes_nor_accumulates(nb_runner, tmp_path):
    """Re-running the cell in a live kernel must land where inline lands.

    Both failure directions are real and both were measured on the way here::

        frozen        CALLS_F  [1] -> [1]       -> [1]        0 calls (no capture)
        accumulating  CALLS_F  [1] -> [1, 1]    -> [1, 1, 1]  1 call each
        correct       CALLS_F  [1] -> [1]       -> [1]        0 calls

    The first two are distinguishable ONLY by the call counter, which is why it
    is asserted rather than the value alone. Accumulation is what happens when
    the global's pre-state is keyed but the checker does not reset it: the
    value the statement produced becomes the next run's key, which misses,
    which produces a third state -- forever.
    """
    cp, cf, ci = _start(nb_runner, tmp_path)
    prev = [_n(cp), _n(cf), _n(ci)]

    for i in (1, 2):
        nb_runner.run_cells(ARMS)
        now = [_n(cp), _n(cf), _n(ci)]
        ran = [a - b for a, b in zip(now, prev)]
        prev = now
        assert _globals(nb_runner) == {"inline": "[1]", "in_callee": "[1]"}, (
            f"re-run {i} diverged from the inline spelling"
        )
        assert ran == [0, 0, 0], (
            f"re-run {i} re-executed {ran} (pure, in-callee, inline); the "
            "in-callee arm caching worse than the pure one means its key never "
            "settles"
        )


@pytest.mark.parametrize("spelling,source", [
    ("assignment", "af = compute_f(1)\n"),
    ("append", "sink = []\nsink.append(compute_f(1))\n"),
    ("nested_call", "print('F', compute_f(1))\n"),
    ("comprehension", "vals = [compute_f(1)]\n"),
])
def test_every_spelling_of_the_call_behaves_the_same(nb_runner, tmp_path, spelling, source):
    """CAS-145: a rule that fires for one spelling and not another is a defect
    this project has already paid for. ``function_global_mutations`` only ever
    walked top-level bare-``Expr`` calls, so every spelling here except a bare
    ``bump()`` was invisible to it.
    """
    cp, cf, ci = (tmp_path / f"{n}.log" for n in ("p", "f", "i"))
    nb_runner.create_notebook([SETUP, _defs(cp, cf, ci), source])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _peek(nb_runner, "globals().get('CALLS_F')") == "[1]"
    cold = _n(cf)

    nb_runner.restart()
    nb_runner.run_all()
    assert _n(cf) - cold == 0, f"{spelling}: nothing was cached, so this proves nothing"
    assert _peek(nb_runner, "globals().get('CALLS_F')") == "[1]", (
        f"{spelling}: the callee's global write was dropped on a hit"
    )


def test_the_statement_stops_caching_but_the_work_does_not(nb_runner, tmp_path):
    """The load-bearing cost claim, and the reason this design is affordable.

    A statement whose callee writes a global is SKIP-CACHED -- it re-executes
    every run, so the write really happens, exactly as the inline spelling of
    the same mutation does. That would be an unacceptable trade if it meant the
    expensive work re-ran too. It does not: sub-statement caching (CAS-243)
    still serves every call inside the statement, so what re-executes is the
    glue.

    Measured as counted executions of BOTH callees in one statement -- the
    global-mutating one and a pure one sharing the statement. A design that
    skip-cached the statement AND lost call-level reuse would show 2 here; one
    that lost only the mutating call's reuse would show 1.
    """
    cf, cp = tmp_path / "f.log", tmp_path / "p.log"
    defs = (
        "import time, os\n"
        "def _tick(p):\n"
        "    fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
        "    os.write(fd, b'X')\n"
        "    os.close(fd)\n"
        "def _busy(ms):\n"
        "    t = time.perf_counter() + ms / 1000.0\n"
        "    while time.perf_counter() < t:\n"
        "        pass\n"
        "SEEN = []\n"
        "def mutating(v):\n"
        "    SEEN.append(v)\n"
        f"    _tick(r'{cf}')\n"
        f"    _busy({_BODY_MS})\n"
        "    return v * 10\n"
        "def pure(v):\n"
        f"    _tick(r'{cp}')\n"
        f"    _busy({_BODY_MS})\n"
        "    return v * 100\n"
    )
    nb_runner.create_notebook([SETUP, defs, "total = mutating(1) + pure(2)\n"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _peek(nb_runner, "globals().get('total')") == "210"
    assert _peek(nb_runner, "globals().get('SEEN')") == "[1]"
    cold = [_n(cf), _n(cp)]
    assert cold == [1, 1]

    nb_runner.restart()
    nb_runner.run_all()

    assert [_n(cf) - cold[0], _n(cp) - cold[1]] == [0, 0], (
        "the statement re-executed AND re-ran its calls; skip-caching the "
        "statement is only affordable because the calls inside it still cache"
    )
    assert _peek(nb_runner, "globals().get('total')") == "210"
    assert _peek(nb_runner, "globals().get('SEEN')") == "[1]", (
        "the callee's global write was lost even though the statement re-ran"
    )


def test_editing_the_callee_still_recomputes(nb_runner, tmp_path):
    """Negative control. Every assertion above would pass just as happily if
    the statement had stopped caching entirely, or if the restore were writing
    a constant."""
    cp, cf, ci = _start(nb_runner, tmp_path)
    cold = _n(cf)
    assert _peek(nb_runner, "globals().get('af')") == "10"

    nb_runner.set_cell_source(DEFS_CELL, _defs(cp, cf, ci, mult=20))
    nb_runner.run_all()

    assert _n(cf) - cold >= 1, "editing the callee did not re-run it"
    assert _peek(nb_runner, "globals().get('af')") == "20", "served a stale value after an edit"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CAS-265 (CAS-260's second half): inside a loop BODY the callee's global write is "
        "still dropped. The capture is deliberately withheld for a control-body "
        "statement -- the loop is ONE unit to the upstream simulation and to the "
        "accumulator machinery, and a per-iteration capture of the global's "
        "ABSOLUTE value clobbers the accumulation (measured: [2], one "
        "iteration's snapshot, instead of [1, 2, 3]). Owning it at the loop "
        "level is the remaining work; when that lands this test passes and the "
        "marker comes off."
    ),
)
def test_a_loop_body_captures_the_callee_global_too(nb_runner, tmp_path):
    cp, cf, ci = (tmp_path / f"{n}.log" for n in ("p", "f", "i"))
    loop_inline = ("for t in [1, 2, 3]:\n"
                   "    CALLS_I.append(t)\n"
                   "    oi.append(compute_i(t))\n")
    loop_callee = ("for t in [1, 2, 3]:\n"
                   "    of.append(compute_f(t))\n")
    nb_runner.create_notebook([
        SETUP, _defs(cp, cf, ci), "oi = []\nof = []\n", loop_inline, loop_callee,
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _globals(nb_runner) == {"inline": "[1, 2, 3]", "in_callee": "[1, 2, 3]"}
    cold = _n(cf)

    nb_runner.restart()
    nb_runner.run_all()
    assert _n(cf) - cold == 0, "the loop was not served from cache, so this proves nothing"
    assert _globals(nb_runner) == {"inline": "[1, 2, 3]", "in_callee": "[1, 2, 3]"}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "CAS-265, the sharper symptom: re-running an EARLIER loop cell leaves a "
        "LATER loop cell's callee-writes in the global. Reported live while "
        "validating the demo. Same root cause as the sibling xfail -- nothing "
        "tracks the global inside a loop body, so there is no reset target -- "
        "but worse than 'the write is skipped on a hit': stale data from a cell "
        "that top-to-bottom had not run yet survives. Flips when CAS-265 lands."
    ),
)
def test_rerunning_an_earlier_loop_discards_a_later_loops_writes(nb_runner, tmp_path):
    """Measured, both spellings in one notebook::

                           inline                  in-callee
        after run_all      [1, 2, 3, 111, 10]      [1, 2, 3, 111, 10]
        re-ran cell A      [1, 2, 3]               [1, 2, 3, 111, 10]   <- wrong

    Inline is the contract and it is unambiguous here: re-running cell A rewinds
    to A's cell-entry state and re-applies only A's appends, which is what a
    clean top-to-bottom run up to A would produce.
    """
    cf, cp, ci = (tmp_path / f"{n}.log" for n in ("f", "p", "i"))
    defs = _defs(cp, cf, ci) + (
        "LOG_I = []\n"
        "def inline_pure(v):\n"
        f"    _busy({_BODY_MS})\n"
        "    return v * 10\n"
    )
    a_inline = ("for x in [1, 2, 3]:\n"
                "    LOG_I.append(x)\n"
                "    oi.append(inline_pure(x))\n")
    a_callee = "for x in [1, 2, 3]:\n    of.append(compute_f(x))\n"
    b_inline = ("for x in [111, 10]:\n"
                "    LOG_I.append(x)\n"
                "    oi.append(inline_pure(x))\n")
    b_callee = "for x in [111, 10]:\n    of.append(compute_f(x))\n"
    nb_runner.create_notebook([
        SETUP, defs, "oi = []\nof = []\n", a_inline, a_callee, b_inline, b_callee,
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    both = {"inline": "[1, 2, 3, 111, 10]", "in_callee": "[1, 2, 3, 111, 10]"}
    live = {"inline": _peek(nb_runner, "globals().get('LOG_I')"),
            "in_callee": _peek(nb_runner, "globals().get('CALLS_F')")}
    assert live == both, f"cold run is not the expected starting point: {live}"

    nb_runner.run_cells([4, 5])          # the EARLIER pair only

    after = {"inline": _peek(nb_runner, "globals().get('LOG_I')"),
             "in_callee": _peek(nb_runner, "globals().get('CALLS_F')")}
    assert after["inline"] == "[1, 2, 3]", (
        f"the inline yardstick itself moved: {after['inline']}"
    )
    assert after["in_callee"] == after["inline"], (
        "re-running the earlier loop left the later loop's callee-writes behind"
    )


def test_oracle_the_harness_measures_real_work(nb_runner, tmp_path):
    """Cash off: every re-run executes everything. Pins the counters, so a zero
    above means reuse rather than a broken instrument."""
    cp, cf, ci = (tmp_path / f"{n}.log" for n in ("p", "f", "i"))
    nb_runner.create_notebook(["import cash\n", _defs(cp, cf, ci), ARM_P, ARM_F, ARM_I])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    cold = [_n(cp), _n(cf), _n(ci)]
    assert cold == [1, 1, 1]

    nb_runner.run_cells(ARMS)
    assert [a - b for a, b in zip([_n(cp), _n(cf), _n(ci)], cold)] == [1, 1, 1], (
        "the oracle did not re-execute; the counters are not measuring real work"
    )
    assert _peek(nb_runner, "globals().get('CALLS_F')") == "[1, 1]"

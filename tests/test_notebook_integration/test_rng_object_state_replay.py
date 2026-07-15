"""Object-held RNG state must be replayed across a cache hit (CAS-90).

``capture_rng_state`` / ``restore_rng_state`` replay the RNG *module globals*
(``random``, ``np.random``, ``torch``) across a cache hit.  They cannot see a
generator the user holds in a variable::

    rng = np.random.default_rng(42)   # cell 1 — cheap, re-executes
    x = expensive(rng.integers(...))  # cell 2 — caches, HITs on re-run
    y = rng.integers(...)             # cell 3 — cheap, re-executes

On the second pass cell 1 rebuilds a *fresh* seed-42 generator, cell 2 HITs so
``x`` is restored without any draw being taken, and cell 3 then draws the FIRST
draw all over again — ``y`` comes back a verbatim duplicate of ``x``.  The
global channel does not have this bug; these tests pin the object channel to
the same contract.

The oracle throughout is a from-scratch ``run_all``: what the notebook prints
on a cold cache is what it must print on a cache pass.
"""

import pytest

pytestmark = pytest.mark.integration

# --- Shared notebook pieces -------------------------------------------------
#
# ``_slow`` pushes the draw statement over the 10 ms 'too cheap to cache' floor
# so cell 2 genuinely caches (and therefore genuinely HITs) without persist
# mode.  The draw stays visible in the statement's RHS, so this is a faithful
# repro rather than one that hides the call behind a helper.
_HELPER = (
    "import numpy as np, random, time\n"
    "def _slow(v):\n"
    "    time.sleep(0.05)\n"
    "    return v\n"
)


def _fmt(name):
    return f"print('{name}=' + ','.join(map(str, {name})))"


# --- 1. Core: np.random.Generator ------------------------------------------


@pytest.mark.timeout(90)
def test_object_generator_state_replayed_across_cache_hit(nb_runner):
    """A cache hit on the draw statement must advance the live generator."""
    nb_runner.create_notebook([
        _HELPER + "rng = np.random.default_rng(42)",
        "x = _slow(rng.integers(0, 100, 3))",
        "y = rng.integers(0, 100, 3)",
        _fmt('x') + "\n" + _fmt('y'),
    ])
    nb_runner.start_kernel()

    # Pass 1 (cold cache) is the oracle.
    nb_runner.run_all()
    out1 = nb_runner.get_output(4)
    assert "x=8,77,65" in out1, out1
    assert "y=43,43,85" in out1, out1

    # Pass 2: cell 2 HITs, cell 3 re-executes against the live generator.
    nb_runner.run_all()
    out2 = nb_runner.get_output(4)
    assert "x=8,77,65" in out2, out2
    assert "y=43,43,85" in out2, (
        "cache hit on the draw statement left the live generator un-advanced, "
        f"so the next draw duplicated x. Expected y=43,43,85, got: {out2!r}"
    )
    assert "y=8,77,65" not in out2, f"y duplicated x: {out2!r}"


# --- 2. Carrier variants ----------------------------------------------------


@pytest.mark.timeout(90)
def test_user_held_randomstate_replayed_across_cache_hit(nb_runner):
    """A user-held np.random.RandomState is a carrier too."""
    nb_runner.create_notebook([
        _HELPER + "rs = np.random.RandomState(42)",
        "x = _slow(rs.randint(0, 100, 3))",
        "y = rs.randint(0, 100, 3)",
        _fmt('x') + "\n" + _fmt('y'),
    ])
    nb_runner.start_kernel()

    nb_runner.run_all()
    out1 = nb_runner.get_output(4)
    assert "x=51,92,14" in out1, out1
    assert "y=71,60,20" in out1, out1

    nb_runner.run_all()
    out2 = nb_runner.get_output(4)
    assert "x=51,92,14" in out2, out2
    assert "y=71,60,20" in out2, (
        f"user-held RandomState not advanced across the hit: {out2!r}"
    )


@pytest.mark.timeout(90)
def test_random_dot_random_instance_replayed_across_cache_hit(nb_runner):
    """A stdlib random.Random instance is a carrier too."""
    nb_runner.create_notebook([
        _HELPER + "r = random.Random(42)",
        "x = _slow([r.randint(0, 100) for _ in range(3)])",
        "y = [r.randint(0, 100) for _ in range(3)]",
        _fmt('x') + "\n" + _fmt('y'),
    ])
    nb_runner.start_kernel()

    nb_runner.run_all()
    out1 = nb_runner.get_output(4)
    assert "x=81,14,3" in out1, out1
    assert "y=94,35,31" in out1, out1

    nb_runner.run_all()
    out2 = nb_runner.get_output(4)
    assert "x=81,14,3" in out2, out2
    assert "y=94,35,31" in out2, (
        f"random.Random instance not advanced across the hit: {out2!r}"
    )


# --- 3. Control: the global channel must not regress ------------------------


@pytest.mark.timeout(90)
def test_global_numpy_random_still_replayed(nb_runner):
    """The already-correct module-global channel keeps working."""
    nb_runner.create_notebook([
        _HELPER + "np.random.seed(42)",
        "x = _slow(np.random.randint(0, 100, 3))",
        "y = np.random.randint(0, 100, 3)",
        _fmt('x') + "\n" + _fmt('y'),
    ])
    nb_runner.start_kernel()

    nb_runner.run_all()
    out1 = nb_runner.get_output(4)
    assert "x=51,92,14" in out1, out1
    assert "y=71,60,20" in out1, out1

    nb_runner.run_all()
    out2 = nb_runner.get_output(4)
    assert "x=51,92,14" in out2, out2
    assert "y=71,60,20" in out2, (
        f"global numpy RNG channel regressed: {out2!r}"
    )


# --- 4. Control: no over-capture --------------------------------------------


@pytest.mark.timeout(90)
def test_non_rng_input_is_not_captured(nb_runner):
    """A statement reading a DataFrame caches and restores exactly as before.

    Doubles as the end-to-end backward-compat proof: because the payload key is
    omitted when a statement reads no RNG carriers, the hit on cell 3 below
    exercises the *missing-key* restore branch on a real kernel — the same code
    path an entry written before CAS-90 takes.
    """
    nb_runner.create_notebook([
        "import pandas as pd, time\n"
        "df = pd.DataFrame({'a': [1, 2, 3]})",
        "def _slow(v):\n"
        "    time.sleep(0.05)\n"
        "    return v\n"
        "total = _slow(int(df['a'].sum()))",
        "print('total=' + str(total))",
    ])
    nb_runner.start_kernel()

    nb_runner.run_all()
    assert "total=6" in nb_runner.get_output(3)

    nb_runner.run_all()
    assert "total=6" in nb_runner.get_output(3)


def test_capture_scoped_to_inputs_and_skips_non_rng_objects():
    """Capture is an isinstance allowlist over the statement's inputs only."""
    import random

    import numpy as np
    import pandas as pd

    from cash.notebook.randomness import capture_object_rng_states

    ns = {
        'df': pd.DataFrame({'a': [1, 2, 3]}),
        'n': 42,
        's': "hello",
        'rng': np.random.default_rng(0),
    }

    # A statement whose inputs are all non-RNG captures nothing at all, so the
    # payload key is omitted and its shape is unchanged.
    assert capture_object_rng_states({'df', 'n', 's'}, ns) == {}

    # Names not in the statement's inputs are never even looked at — this is
    # what bounds the cost (no full user_ns walk).
    assert capture_object_rng_states({'df'}, ns) == {}
    assert 'rng' in capture_object_rng_states({'rng', 'df'}, ns)

    # Missing names are tolerated.
    assert capture_object_rng_states({'nope'}, ns) == {}

    # The module-global singletons are owned by the global channel, not this
    # one — capturing them here would let a stale alias fight the global state.
    assert capture_object_rng_states(
        {'g'}, {'g': np.random.mtrand._rand},
    ) == {}
    assert capture_object_rng_states(
        {'g'}, {'g': random._inst},
    ) == {}


def test_foreign_and_unpicklable_carriers_are_skipped():
    """A carrier whose state can't be read is dropped, not raised."""
    import random

    import numpy as np

    from cash.notebook.randomness import (
        capture_object_rng_states,
        restore_object_rng_states,
    )

    # random.SystemRandom subclasses random.Random but getstate() raises
    # NotImplementedError — it must be skipped silently at capture time.
    assert capture_object_rng_states(
        {'sr'}, {'sr': random.SystemRandom()},
    ) == {}

    # On the restore side, a state the live bit generator rejects (a foreign or
    # mismatched bit generator) must be dropped, not propagated to the user.
    rng = np.random.default_rng(0)
    states = capture_object_rng_states({'rng'}, {'rng': rng})
    states['rng']['state'] = {'bit_generator': 'NotAPCG64', 'state': 'garbage'}
    restore_object_rng_states(states, {'rng': rng})  # must not raise


# --- 5. Backward compatibility ----------------------------------------------


def test_payload_without_object_rng_field_restores_cleanly():
    """Entries written before CAS-90 lack the field and must load unchanged."""
    import numpy as np

    from cash.notebook.randomness import restore_object_rng_states

    rng = np.random.default_rng(42)
    before = rng.bit_generator.state

    # An old payload simply has no 'rng_object_states' key; the restore path
    # reads it with .get(), yielding None.
    old_payload = {'variables': {'x': 1}, 'stdout': '', 'rng_state': {}}
    restore_object_rng_states(old_payload.get('rng_object_states'), {'rng': rng})

    assert rng.bit_generator.state == before, "no-op restore perturbed the RNG"

    # Empty dict is a no-op too.
    restore_object_rng_states({}, {'rng': rng})
    assert rng.bit_generator.state == before


def test_restore_guards_on_presence_and_type_match():
    """A name that vanished or now holds a different kind is skipped."""
    import numpy as np

    from cash.notebook.randomness import (
        capture_object_rng_states,
        restore_object_rng_states,
    )

    rng = np.random.default_rng(42)
    rng.integers(0, 100, 3)
    states = capture_object_rng_states({'rng'}, {'rng': rng})
    assert states

    # Name absent from the namespace -> skipped, no raise.
    restore_object_rng_states(states, {})

    # Name now holds a non-carrier -> type match fails, skipped, no raise.
    restore_object_rng_states(states, {'rng': "not an rng"})

    # Name holds a different carrier kind -> skipped rather than forced.
    rs = np.random.RandomState(0)
    rs_state_before = rs.get_state()
    restore_object_rng_states(states, {'rng': rs})
    assert rs.get_state()[1].tolist() == rs_state_before[1].tolist()


def test_aliased_names_resolve_to_the_same_object():
    """`rng2 = rng` is the same object; setting state by name is idempotent."""
    import numpy as np

    from cash.notebook.randomness import (
        capture_object_rng_states,
        restore_object_rng_states,
    )

    rng = np.random.default_rng(42)
    rng2 = rng
    rng.integers(0, 100, 3)

    ns = {'rng': rng, 'rng2': rng2}
    states = capture_object_rng_states({'rng', 'rng2'}, ns)
    assert set(states) == {'rng', 'rng2'}

    post = rng.bit_generator.state
    rng.integers(0, 100, 3)  # drift the live object
    restore_object_rng_states(states, ns)

    # Both names wrote the same state onto the same object: idempotent.
    assert rng.bit_generator.state == post
    assert rng2.bit_generator.state == post

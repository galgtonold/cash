"""CAS-160 regression: `# @cash:persist` on a loop that incrementally builds a
frame must NOT snapshot every intermediate column-width without bound.

Before the fix, 200k rows x 25 column-adds wrote 520,221,515 B across 53 files
for a 40 MB final frame -- 13.01x amplification, no warning, and a per-iteration
cost climbing 0.04s -> 0.16s as each iteration re-serialised an ever-wider
frame. CAS-142's caps could not see it: its per-object refusal compares ONE
value against half the tier cap (40 MB vs >=4 GiB) and its evict-after-write
warning needs the total to exceed the cap (520 MB vs >=8 GiB), so neither ever
fired. The guard added for CAS-160 tracks the missing dimension -- cumulative
writes for ONE statement -- and stops value-persisting once that total is out of
proportion to the value, warning once.

Witness is EXTERNAL: on-disk bytes under the notebook work dir, measured with
os.walk/stat from the test process. The badge is not trusted.

Each column-add is made to clear cash's 10 ms "too cheap to cache" floor via a
sleeping helper, so the persist machinery genuinely engages (an all-cheap loop
would be skipped for an unrelated reason and prove nothing).
"""
import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

ROWS = 200_000
COLS = 25
FINAL_BYTES = ROWS * COLS * 8                                   # 40 MB
ALL_SNAPSHOT_BYTES = ROWS * 8 * (COLS * (COLS + 1) // 2)        # 520 MB


def _ascii(s):
    return s.encode("ascii", "replace").decode("ascii")


def _cache_bytes(root):
    total = 0
    files = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            try:
                total += os.stat(os.path.join(dirpath, fn)).st_size
                files += 1
            except OSError:
                pass
    return total, files


def test_persist_on_incremental_frame_loop_cache_size(nb_runner, tmp_path):
    nb_runner.create_notebook([
        "import cash\n%cash_on\n%cash_badge print",
        "import numpy as np\nimport pandas as pd\nimport time",
        f"ROWS = {ROWS}\nCOLS = {COLS}\nrng = np.random.default_rng(0)",
        # Slow enough per column that each add clears the 10 ms floor.
        "def slow_col(n):\n    time.sleep(0.03)\n    return rng.random(n)",
        # The ticket's shape: persist on a loop that grows a frame a column at a time.
        "# @cash:persist\n"
        "df = pd.DataFrame(index=range(ROWS))\n"
        "for i in range(COLS):\n"
        "    df[f'c{i}'] = slow_col(ROWS)",
        "print('shape=', df.shape)\nprint('nbytes=', int(df.values.nbytes))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    out6 = nb_runner.get_output(6)
    assert f"shape= ({ROWS}, {COLS})" in out6, out6

    total, files = _cache_bytes(tmp_path)
    raw5 = _ascii(nb_runner.get_raw_output(5))

    print("\n=== CAS-160 measurement ===")
    print(f"final frame bytes      : {FINAL_BYTES:,}")
    print(f"if every add snapshots : {ALL_SNAPSHOT_BYTES:,}")
    print(f"on-disk cache bytes    : {total:,}  ({files} files)")
    print(f"amplification vs final : {total / FINAL_BYTES:.2f}x")
    print("--- persist cell raw output (first 6000 chars) ---")
    print(raw5[:6000])

    # Negative control: at the TOP of the cell the directive binds only to the
    # first statement (the empty-frame seed), so nothing amplifies here.
    # (Statement scoping is deliberate -- CAS-189 keeps `persist` bound to one
    # statement precisely so it cannot spread this amplification cell-wide.)
    assert total < FINAL_BYTES, (
        f"cell-top persist unexpectedly persisted the frame: {total:,} bytes"
    )
    # Nothing amplified, so the CAS-160 guard must stay silent. This is the
    # false-positive side of the guard: it must not shout at a healthy notebook.
    assert "runs in a loop and has already cached" not in raw5, (
        f"amplification warning fired with no amplification:\n{raw5[:3000]}"
    )


def _run_variant(nb_runner, tmp_path, loop_cell, label):
    nb_runner.create_notebook([
        "import cash\n%cash_on\n%cash_badge print",
        "import numpy as np\nimport pandas as pd\nimport time",
        f"ROWS = {ROWS}\nCOLS = {COLS}\nrng = np.random.default_rng(0)",
        "def slow_col(n):\n    time.sleep(0.03)\n    return rng.random(n)",
        loop_cell,
        "print('shape=', df.shape)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert f"shape= ({ROWS}, {COLS})" in nb_runner.get_output(6), nb_runner.get_output(6)

    total, files = _cache_bytes(tmp_path)
    raw5 = _ascii(nb_runner.get_raw_output(5))
    print(f"\n=== CAS-160 {label} ===")
    print(f"final frame bytes      : {FINAL_BYTES:,}")
    print(f"if every add snapshots : {ALL_SNAPSHOT_BYTES:,}")
    print(f"on-disk cache bytes    : {total:,}  ({files} files)")
    print(f"amplification vs final : {total / FINAL_BYTES:.2f}x")
    print(raw5[:6000])
    return total, raw5


def _assert_bounded_and_warned(total, raw, label):
    """The two things CAS-160 promises: the cache stays proportional to the
    final value, and the user is told once why persisting stopped."""
    # Bounded. The guard lets the loop run until one statement has written its
    # floor (64 MiB) and that total dwarfs the current value, then latches off,
    # so the cache lands a little over the floor regardless of how many more
    # iterations follow. Measured 72,089,669 B (1.80x) vs 520,221,515 (13.01x)
    # before the fix. The bound is what matters, not the exact landing point.
    assert total < 3 * FINAL_BYTES, (
        f"{label}: persist still amplifies -- {total:,} B "
        f"({total / FINAL_BYTES:.2f}x the {FINAL_BYTES:,} B final frame)"
    )
    # ...and loudly, once. Silence here is the actual CAS-160 bug: the user's
    # disk filled with no signal at all.
    assert "runs in a loop and has already cached" in raw, (
        f"{label}: cache was bounded but the user was never told why. "
        f"Cell output:\n{raw[:3000]}"
    )
    assert raw.count("runs in a loop and has already cached") == 1, (
        f"{label}: the amplification warning repeated instead of firing once"
    )


def test_persist_directive_on_the_for_line(nb_runner, tmp_path):
    """Directive scoped to the loop unit itself (the placement the accumulator
    suite documents as correct). Every iteration is a persist target, so the
    guard must cap the intermediate snapshots and say so."""
    total, raw = _run_variant(
        nb_runner, tmp_path,
        "df = pd.DataFrame(index=range(ROWS))\n"
        "# @cash:persist\n"
        "for i in range(COLS):\n"
        "    df[f'c{i}'] = slow_col(ROWS)",
        "variant B: persist on the FOR line",
    )
    _assert_bounded_and_warned(total, raw, "persist on the FOR line")


def test_persist_directive_inside_the_loop_body(nb_runner, tmp_path):
    """Worst case: the directive sits on the column-add itself, so every
    iteration is a persist target."""
    total, raw = _run_variant(
        nb_runner, tmp_path,
        "df = pd.DataFrame(index=range(ROWS))\n"
        "for i in range(COLS):\n"
        "    # @cash:persist\n"
        "    df[f'c{i}'] = slow_col(ROWS)",
        "variant C: persist INSIDE the loop body",
    )
    _assert_bounded_and_warned(total, raw, "persist INSIDE the loop body")

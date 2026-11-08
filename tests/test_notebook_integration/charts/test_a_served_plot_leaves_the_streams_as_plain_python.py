"""After a plot that draws random numbers is shown from the cache, every
later cell sees what plain Python (restart + Run All) gives.

Each test compares the kernel against the same cells run top to bottom in a
fresh interpreter (`_plain`). A plot or a statement that seeds a stream from
the system's entropy, anywhere in it, is never served: in plain Python the
draws below it differ on every run.
"""

from __future__ import annotations

import json

import pytest

from tests._scripts import run_python
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")

HEAD = (
    "%matplotlib inline\nimport os, random, time\nimport numpy as np\nimport matplotlib.pyplot as plt\n"
    "from functools import cache\nLOG = r'{log}'"
)
HELPERS = (
    "def count():\n"
    "    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    "\n"
    "def noisy_plot(n):\n"
    "    count()\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    ys = [random.random() for _ in range(n)]\n"
    "    zs = np.random.rand(n)\n"
    "    plt.figure(figsize=(2, 2))\n"
    "    plt.plot(ys, zs, 'o')\n"
    "    return round(sum(ys) + float(zs.sum()), 12)\n"
    "\n"
    "def seeding_plot(n):\n"
    "    count()\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    random.seed(5)\n"
    "    np.random.seed(5)\n"
    "    ys = [random.random() for _ in range(n)]\n"
    "    zs = np.random.rand(n)\n"
    "    plt.figure(figsize=(2, 2))\n"
    "    plt.plot(ys, zs, 'o')\n"
    "    return round(sum(ys) + float(zs.sum()), 12)\n"
    "\n"
    "def gen_plot(g):\n"
    "    count()\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    ys = [g.random() for _ in range(3)]\n"
    "    plt.figure(figsize=(2, 2))\n"
    "    plt.plot(ys, 'o')\n"
    "    return round(sum(ys), 12)\n"
    "\n"
    "def slow_draw():\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return random.random(), float(np.random.rand())\n"
    "\n"
    "@cache\n"
    "def cities(n):\n"
    "    random.seed(n)\n"
    "    return tuple(random.random() for _ in range(n))\n"
    "\n"
    "def cities_plot(n):\n"
    "    count()\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    cs = cities(n)\n"
    "    noise = random.random()\n"
    "    plt.figure(figsize=(2, 2))\n"
    "    plt.plot(cs, 'o')\n"
    "    return round(sum(cs) + noise, 12)\n"
)
SEED = "random.seed(1)\nnp.random.seed(1)"
AFTER = "after = (random.random(), float(np.random.rand()))"
WATCH = ("after", "(hash(random.getstate()), int(np.random.get_state()[2]))")


def _plain(tmp_path, cells: list[str], exprs: tuple[str, ...]) -> list:
    """*exprs* after *cells* run top to bottom in a fresh interpreter."""
    lines = ["import matplotlib", "matplotlib.use('Agg')", "ns = {}"]
    for i, cell in enumerate(cells):
        src = "\n".join(line for line in cell.splitlines() if not line.startswith("%"))
        lines.append(f"exec(compile({src!r}, 'cell{i}', 'exec'), ns)")
        if i == 0:
            lines.append(f"ns['LOG'] = {str(tmp_path / 'plain.log')!r}")
    lines.append(f"import json; print(json.dumps([repr(eval(e, ns)) for e in {list(exprs)!r}]))")
    script = tmp_path / "plain.py"
    script.write_text("\n".join(lines), encoding="utf-8")
    out = run_python(script, cwd=tmp_path)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _kernel(nb_runner, exprs) -> list:
    return [nb_runner.peek(e) for e in exprs]


def _calls(log) -> int:
    return len(log.read_text(encoding="utf-8")) if log.exists() else 0


def _value(cell) -> str:
    for o in cell.get("outputs", []):
        if o["output_type"] == "execute_result":
            return o["data"].get("text/plain", "")
    return ""


def _setup(nb_runner, tmp_path, *cells):
    log = tmp_path / "calls.log"
    all_cells = [HEAD.format(log=log), HELPERS, *cells]
    nb_runner.create_notebook(all_cells)
    nb_runner.start_kernel()
    return log, all_cells


def _cells(nb_runner) -> list[str]:
    return [c.source for c in nb_runner.nb.cells]


def _check(nb_runner, tmp_path, exprs=WATCH, plot_cell=None):
    want = _plain(tmp_path, _cells(nb_runner), exprs)
    assert _kernel(nb_runner, exprs) == want
    if plot_cell is not None:
        shown = _value(nb_runner.get_cell(plot_cell))
        plain_value = _plain(tmp_path, _cells(nb_runner)[: plot_cell - 1], (_cells(nb_runner)[plot_cell - 1],))
        assert shown == plain_value[0], "the plot showed another value than plain Python"


def test_served_plot_then_a_draw(nb_runner, tmp_path):
    log, _ = _setup(nb_runner, tmp_path, SEED, "noisy_plot(3)", AFTER)
    nb_runner.run_all()
    _check(nb_runner, tmp_path, plot_cell=4)
    nb_runner.run_all()
    assert _calls(log) == 1, "the unchanged plot was drawn again"
    _check(nb_runner, tmp_path, plot_cell=4)
    nb_runner.run_all()
    assert _calls(log) == 1
    _check(nb_runner, tmp_path, plot_cell=4)


def test_two_served_plots_in_a_row(nb_runner, tmp_path):
    log, _ = _setup(nb_runner, tmp_path, SEED, "noisy_plot(3)", "noisy_plot(2)", AFTER)
    nb_runner.run_all()
    nb_runner.run_all()
    assert _calls(log) == 2, "the unchanged plots were drawn again"
    _check(nb_runner, tmp_path, plot_cell=5)
    # An edit of the first plot's draws: the second must follow.
    nb_runner.set_cell_source(4, "noisy_plot(4)")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, plot_cell=5)


def test_edit_of_the_seed_above(nb_runner, tmp_path):
    log, _ = _setup(nb_runner, tmp_path, SEED, "noisy_plot(3)", AFTER)
    nb_runner.run_all()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, "random.seed(2)\nnp.random.seed(2)")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, plot_cell=4)
    nb_runner.set_cell_source(3, SEED)
    nb_runner.run_all()
    _check(nb_runner, tmp_path, plot_cell=4)


def test_edit_of_the_number_of_draws_above(nb_runner, tmp_path):
    log, _ = _setup(nb_runner, tmp_path, SEED, "pre = random.random()", "noisy_plot(3)", AFTER)
    nb_runner.run_all()
    nb_runner.run_all()
    assert _calls(log) == 1
    nb_runner.set_cell_source(4, "pre = (random.random(), random.random())")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, plot_cell=5)
    nb_runner.set_cell_source(4, "pre = random.random()")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, plot_cell=5)


def test_a_restored_draw_above_then_a_served_plot(nb_runner, tmp_path):
    log, _ = _setup(nb_runner, tmp_path, SEED, "pre = slow_draw()", "noisy_plot(3)", AFTER)
    nb_runner.run_all()
    nb_runner.run_all()
    assert _calls(log) == 1, "the unchanged plot was drawn again"
    _check(nb_runner, tmp_path, plot_cell=5)
    nb_runner.set_cell_source(3, "random.seed(3)\nnp.random.seed(3)")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, plot_cell=5)


@pytest.mark.fresh_kernel
def test_restart_and_run_all_from_disk(nb_runner, tmp_path):
    log, _ = _setup(
        nb_runner,
        tmp_path,
        SEED,
        "pre = slow_draw()",
        "noisy_plot(3)",
        AFTER,
        "seeding_plot(2)",
        AFTER.replace("after", "after2"),
    )
    nb_runner.run_all()
    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert _calls(log) == 2, "after a restart the unchanged plots were drawn again"
    _check(nb_runner, tmp_path, exprs=(*WATCH, "after2"), plot_cell=5)
    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.set_cell_source(3, "random.seed(4)\nnp.random.seed(4)")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, exprs=(*WATCH, "after2"), plot_cell=5)


def test_a_plot_that_reseeds_then_a_draw_without_seeding(nb_runner, tmp_path):
    log, _ = _setup(nb_runner, tmp_path, "pre = 1", "seeding_plot(3)", AFTER)
    nb_runner.run_all()
    nb_runner.run_all()
    assert _calls(log) == 1, "the unchanged reseeding plot was drawn again"
    _check(nb_runner, tmp_path, plot_cell=4)
    nb_runner.set_cell_source(5, "after = (random.random(), random.random(), float(np.random.rand()))")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, plot_cell=4)


def test_a_plot_that_reseeds_in_a_cell_that_draws_after_it(nb_runner, tmp_path):
    log, _ = _setup(
        nb_runner, tmp_path, SEED, "v = seeding_plot(3)\nplt.show()\nafter = (random.random(), float(np.random.rand()))"
    )
    nb_runner.run_all()
    nb_runner.run_all()
    _check(nb_runner, tmp_path)


def test_a_numpy_generator_in_a_variable(nb_runner, tmp_path):
    log, _ = _setup(nb_runner, tmp_path, "g = np.random.default_rng(0)", "gen_plot(g)", "after = g.random()")
    nb_runner.run_all()
    nb_runner.run_all()
    _check(nb_runner, tmp_path, exprs=("after",), plot_cell=4)


def test_a_random_instance_in_a_variable(nb_runner, tmp_path):
    log, _ = _setup(nb_runner, tmp_path, "g = random.Random(0)", "gen_plot(g)", "after = g.random()")
    nb_runner.run_all()
    nb_runner.run_all()
    _check(nb_runner, tmp_path, exprs=("after",), plot_cell=4)


def test_a_generator_seeded_in_the_plot_cell(nb_runner, tmp_path):
    log, _ = _setup(nb_runner, tmp_path, "g = np.random.default_rng(0)\ngen_plot(g)", "after = g.random()")
    nb_runner.run_all()
    nb_runner.run_all()
    _check(nb_runner, tmp_path, exprs=("after",))


def _restart(nb_runner):
    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner._init_cash()


@pytest.mark.fresh_kernel
def test_a_memo_filled_above_before_a_served_plot(nb_runner, tmp_path):
    """The plot's first run filled the memo (seeding the stream); after a
    restart a cell above fills it and a draw follows, so in plain Python the
    plot's memo call no longer seeds and its noise is another draw."""
    log, _ = _setup(nb_runner, tmp_path, SEED, "warm = 0", "z = 0", "cities_plot(4)", AFTER)
    nb_runner.run_all()
    _restart(nb_runner)
    nb_runner.run_all()
    assert _calls(log) == 1, "control: the unchanged plot was served after a restart"
    _check(nb_runner, tmp_path, exprs=(*WATCH, "cities.cache_info().currsize"), plot_cell=6)
    _restart(nb_runner)
    nb_runner.set_cell_source(4, "warm = cities(4)")
    nb_runner.set_cell_source(5, "z = random.random()")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, exprs=(*WATCH, "cities.cache_info().currsize"), plot_cell=6)


@pytest.mark.fresh_kernel
def test_a_memo_with_more_answers_than_at_store_time(nb_runner, tmp_path):
    log, _ = _setup(nb_runner, tmp_path, SEED, "warm = 0", "cities_plot(4)", AFTER)
    nb_runner.run_all()
    _restart(nb_runner)
    nb_runner.set_cell_source(4, "warm = cities(2)")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, exprs=(*WATCH, "cities.cache_info().currsize"), plot_cell=5)


@pytest.mark.fresh_kernel
def test_a_memo_read_at_store_time_then_filled_by_the_plot(nb_runner, tmp_path):
    """Stored while a cell above had filled the memo (the plot only read it);
    after a restart that cell no longer fills it, so the plot's call seeds."""
    log, _ = _setup(nb_runner, tmp_path, SEED, "warm = cities(4)", "z = random.random()", "cities_plot(4)", AFTER)
    nb_runner.run_all()
    _restart(nb_runner)
    nb_runner.set_cell_source(4, "warm = 0")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, exprs=(*WATCH, "cities.cache_info().currsize"), plot_cell=6)


def test_a_memo_cleared_above(nb_runner, tmp_path):
    log, _ = _setup(nb_runner, tmp_path, SEED, "warm = cities(4)", "cities_plot(4)", AFTER)
    nb_runner.run_all()
    nb_runner.run_all()
    _check(nb_runner, tmp_path, plot_cell=5)
    nb_runner.set_cell_source(4, "cities.cache_clear()\nwarm = 0")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, exprs=(*WATCH, "cities.cache_info().currsize"), plot_cell=5)


SLOW_AFTER = (
    "def slow_after():\n"
    "    fd = os.open(LOG + '.after', os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    return random.random(), float(np.random.rand())\n"
)


def _after_calls(tmp_path) -> int:
    return _calls(tmp_path / "calls.log.after")


def test_a_stored_draw_below_a_served_plot_is_reused(nb_runner, tmp_path):
    log, _ = _setup(nb_runner, tmp_path, SLOW_AFTER, SEED, "noisy_plot(3)", "after = slow_after()")
    nb_runner.run_all()
    nb_runner.run_all()
    assert _calls(log) == 1
    _check(nb_runner, tmp_path, exprs=("after",))
    nb_runner.run_all()
    _check(nb_runner, tmp_path, exprs=("after",))
    assert _after_calls(tmp_path) == 1, "the stored draw below the served plot ran again"


@pytest.mark.fresh_kernel
def test_a_stored_draw_below_a_served_plot_is_reused_after_a_restart(nb_runner, tmp_path):
    log, _ = _setup(nb_runner, tmp_path, SLOW_AFTER, SEED, "noisy_plot(3)", "after = slow_after()")
    nb_runner.run_all()
    _restart(nb_runner)
    nb_runner.run_all()
    assert _calls(log) == 1
    _check(nb_runner, tmp_path, exprs=("after",))
    assert _after_calls(tmp_path) == 1, "the stored draw below the served plot ran again after a restart"


def test_a_stored_draw_below_a_served_plot_follows_an_edit_above(nb_runner, tmp_path):
    log, _ = _setup(nb_runner, tmp_path, SLOW_AFTER, SEED, "noisy_plot(3)", "after = slow_after()")
    nb_runner.run_all()
    nb_runner.run_all()
    nb_runner.set_cell_source(5, "noisy_plot(4)")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, exprs=("after",))
    nb_runner.set_cell_source(5, "noisy_plot(3)")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, exprs=("after",))


def test_a_seed_in_the_plot_cell_then_a_draw_in_the_same_cell(nb_runner, tmp_path):
    log, _ = _setup(
        nb_runner,
        tmp_path,
        "random.seed(1)\nnp.random.seed(1)\nnoisy_plot(3)\nplt.show()\nafter = (random.random(), float(np.random.rand()))",
        "after2 = random.random()",
    )
    nb_runner.run_all()
    nb_runner.run_all()
    _check(nb_runner, tmp_path, exprs=(*WATCH, "after2"))
    nb_runner.set_cell_source(
        3,
        "random.seed(2)\nnp.random.seed(2)\nnoisy_plot(3)\nplt.show()\nafter = (random.random(), float(np.random.rand()))",
    )
    nb_runner.run_all()
    _check(nb_runner, tmp_path, exprs=(*WATCH, "after2"))


def test_a_seed_set_by_a_helper_above(nb_runner, tmp_path):
    log, _ = _setup(
        nb_runner,
        tmp_path,
        "def setup(s):\n    random.seed(s)\n    np.random.seed(s)",
        "setup(1)",
        "noisy_plot(3)",
        AFTER,
    )
    nb_runner.run_all()
    nb_runner.run_all()
    _check(nb_runner, tmp_path, plot_cell=5)
    nb_runner.set_cell_source(4, "setup(2)")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, plot_cell=5)
    nb_runner.set_cell_source(3, "def setup(s):\n    random.seed(s + 1)\n    np.random.seed(s)")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, plot_cell=5)


def test_two_plot_statements_in_one_cell(nb_runner, tmp_path):
    log, _ = _setup(
        nb_runner,
        tmp_path,
        SEED,
        "noisy_plot(3)\nplt.show()\nseeding_plot(2)\nplt.show()\nnoisy_plot(1)\nplt.show()",
        AFTER,
    )
    nb_runner.run_all()
    nb_runner.run_all()
    _check(nb_runner, tmp_path)


def test_a_draw_cell_between_two_served_plots_rewritten(nb_runner, tmp_path):
    log, _ = _setup(nb_runner, tmp_path, SEED, "noisy_plot(3)", "mid = random.random()", "noisy_plot(2)", AFTER)
    nb_runner.run_all()
    nb_runner.run_all()
    _check(nb_runner, tmp_path, exprs=(*WATCH, "mid"), plot_cell=6)
    nb_runner.set_cell_source(5, "mid = float(np.random.rand())")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, exprs=(*WATCH, "mid"), plot_cell=6)


GLOBAL_GEN = (
    "def global_gen_plot():\n"
    "    count()\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    ys = [G.random() for _ in range(3)]\n"
    "    plt.figure(figsize=(2, 2))\n"
    "    plt.plot(ys, 'o')\n"
    "    return round(sum(ys), 12)\n"
)


@pytest.mark.parametrize("make", ["random.Random(0)", "np.random.RandomState(0)", "np.random.default_rng(0)"])
def test_a_generator_a_helper_reads_as_a_global(nb_runner, tmp_path, make):
    log, _ = _setup(nb_runner, tmp_path, GLOBAL_GEN, f"G = {make}", "global_gen_plot()", "after = float(G.random())")
    nb_runner.run_all()
    nb_runner.run_all()
    _check(nb_runner, tmp_path, exprs=("after",), plot_cell=5)


def test_a_seeding_plot_above_a_reading_plot_edited(nb_runner, tmp_path):
    log, _ = _setup(nb_runner, tmp_path, "seeding_plot(3)", "noisy_plot(2)", AFTER)
    nb_runner.run_all()
    nb_runner.run_all()
    _check(nb_runner, tmp_path, plot_cell=4)
    nb_runner.set_cell_source(3, "seeding_plot(2)")
    nb_runner.run_all()
    _check(nb_runner, tmp_path, plot_cell=4)


def test_a_visible_draw_in_the_plot_statement(nb_runner, tmp_path):
    cell = (
        "def slow_line(ys):\n"
        "    count()\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    plt.figure(figsize=(2, 2))\n"
        "    plt.plot(ys, 'o')\n"
        "    return round(sum(ys), 12)"
    )
    log, _ = _setup(nb_runner, tmp_path, cell, SEED, "slow_line([random.random() for _ in range(3)])", AFTER)
    nb_runner.run_all()
    nb_runner.run_all()
    _check(nb_runner, tmp_path, plot_cell=5)


ENTROPY_AFTER = (
    "def entropy_plot(n):\n"
    "    count()\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    "    random.seed(5)\n"
    "    ys = [random.random() for _ in range(n)]\n"
    "    random.seed()\n"
    "    plt.figure(figsize=(2, 2))\n"
    "    plt.plot(ys, 'o')\n"
    "    return round(sum(ys), 12)\n"
)


def test_a_plot_that_reseeds_from_entropy_after_its_seeded_draws_is_drawn_again(nb_runner, tmp_path):
    """The docs: a plot that seeds a generator from the system's entropy is
    drawn again. Plain Python leaves the stream at a fresh entropy state
    each run, so the draw below differs run to run."""
    log, _ = _setup(nb_runner, tmp_path, ENTROPY_AFTER, "entropy_plot(3)", "after = random.random()")
    nb_runner.run_all()
    first = nb_runner.peek("after")
    nb_runner.run_all()
    second = nb_runner.peek("after")
    assert _calls(log) == 2, "a plot that reseeds from entropy was served"
    assert first != second, "the draw below an entropy reseed repeated"


@pytest.mark.parametrize("seed", ["os.urandom(8)", "time.time_ns()"])
def test_a_plot_that_seeds_from_the_system_first_is_drawn_again(nb_runner, tmp_path, seed):
    helper = ENTROPY_AFTER.replace("random.seed(5)", f"random.seed({seed})").replace("    random.seed()\n", "")
    log, _ = _setup(nb_runner, tmp_path, helper, "entropy_plot(3)", "after = random.random()")
    nb_runner.run_all()
    first = nb_runner.peek("after")
    nb_runner.run_all()
    second = nb_runner.peek("after")
    assert _calls(log) == 2, "a plot that seeds from the system's entropy was served"
    assert first != second, "the draw below an entropy seed repeated"


def test_control_a_stored_statement_that_reseeds_from_entropy_after_its_seeded_draws(nb_runner, tmp_path):
    """Control, no plot: the same helper in an ordinary stored statement."""
    helper = ENTROPY_AFTER.replace("    plt.figure(figsize=(2, 2))\n    plt.plot(ys, 'o')\n", "")
    log, _ = _setup(nb_runner, tmp_path, helper, "v = entropy_plot(3)", "after = random.random()")
    nb_runner.run_all()
    first = nb_runner.peek("after")
    nb_runner.run_all()
    second = nb_runner.peek("after")
    assert first != second, "the draw below an entropy reseed repeated"


def test_a_numpy_plot_that_reseeds_from_entropy_after_its_seeded_draws(nb_runner, tmp_path):
    helper = (
        ENTROPY_AFTER.replace("random.seed(5)", "np.random.seed(5)")
        .replace("random.seed()", "np.random.seed()")
        .replace("random.random()", "float(np.random.rand())")
    )
    log, _ = _setup(nb_runner, tmp_path, helper, "entropy_plot(3)", "after = float(np.random.rand())")
    nb_runner.run_all()
    first = nb_runner.peek("after")
    nb_runner.run_all()
    second = nb_runner.peek("after")
    assert _calls(log) == 2, "a plot that reseeds numpy from entropy was served"
    assert first != second, "the draw below an entropy reseed repeated"

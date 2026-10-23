# Plain checker

Runs a small repro ("shape") under cash and under plain Python, and reports
every place the two differ. Run it on a shape before asking for review; a
reviewer runs the same shapes instead of writing a probe by hand.

## What it runs

- **Notebook shapes:** the cells go through Run All three times, a kernel
  restart, then Run All twice, in a real kernel. Cash is turned on again
  after the restart. The same sequence then runs in a plain kernel.
- **Decorator shapes:** setup code defines functions decorated with the bare
  name `@cache`. The steps run three times in one process, then twice in a
  new process that uses the same cache folder (the restart). Under cash,
  `cache` is `cash.cache`. In plain Python it does nothing. The setup is
  written to a real module (`shape_setup.py`), so cash can read its source
  as it reads a user's file.

Each cash run is repeated in three configs:

| Config | What it sets |
|---|---|
| `trace` | nothing (the default) |
| `notrace` | `CASH_TRACE=0` |
| `floors0` | `CASH_MIN_EXECUTION_TIME_TO_CACHE_SECONDS=0`, `CASH_CALL_COST_FLOOR_SECONDS=0`, `call_unit._COST_FLOOR_S = 0` |

## What it compares

- Each cell's output: stdout, the shown value and any error. stderr is
  left out because cash's warnings go there, and so is display data
  because the badge goes there.
- After each Run All, the variables the shape names. Without `names`, it
  compares every top-level name the cells assign.
- For decorator shapes, each step's value (when the step is a bare
  expression) and what the step printed.

Arrays and frames are compared by content. Other values are compared by
`repr`, with object addresses removed.

Each shape gets a verdict:

- **OK** means nothing differs.
- **BAD** means a value or output differs from plain Python. The exit
  status is then 1.
- **AS-RESTART** means a variable the shape lists in `module_writes`
  differs from the same kernel's plain run, but equals what a restart and a
  Run All give. This is accepted. For module state a statement writes (a
  counter a function bumps), the reference is a restart followed by a Run
  All, because cash puts the write back on a hit (see "Setting state on one
  of your modules" in docs/known-limitations.md).
- **REBUILT** means any other variable differs in the same way: its value
  equals a fresh kernel's Run All of the same cells. This is by design when
  cash rebuilds a generator that a variable holds (see "Randomness"). A
  stale value looks the same, though, so read each one. A real stale value
  matches neither run and shows as BAD.

An error in a cell is compared as output, and the run goes on.

The report also lists cash's status for every statement, per run and
config. For a decorator step, the status is `ran` when a decorated
function's body ran, else `-`.

## Use it

From the repository root, with an absolute `PYTHONPATH` so the kernels
import this tree's cash:

```bash
PYTHONPATH=$PWD/src:$PWD python -m tools.plain_checker                 # every shape in shapes/ but smoke.py
PYTHONPATH=$PWD/src:$PWD python -m tools.plain_checker my_shapes.py -k memo --configs trace -j 3
PYTHONPATH=$PWD/src:$PWD python -m tools.plain_checker --list
```

Options:

- `--statuses` prints the statuses for passing shapes too.
- `--json FILE` also writes the reports to a file.
- `--work DIR` keeps the run folders.

The exit status is 1 when any shape differs from plain Python.

From pytest:

```python
from tools.plain_checker import check, decorator, notebook


def test_my_shape(tmp_path):
    shape = notebook("my-shape", ["xs = [1, 2]", "total = sum(xs)", "xs.append(3)"])
    check(shape, tmp_path).assert_ok()  # or check(shape, tmp_path, ["trace"])
```

## Writing shapes

A shape file is a Python file that defines `SHAPES`:

```python
from tools.plain_checker import decorator, notebook

SHAPES = [
    notebook(
        "list-append",
        cells=["xs = [1, 2, 3]", "total = sum(xs)", "xs.append(4)"],
        names=["total"],  # optional
        edits={3: {0: "xs = [10, 20]"}},  # optional: before run 3 (the first after the restart), cell 0 changes
        source="where the repro came from",
    ),
    decorator(
        "mutated-argument",
        setup="@cache\ndef total(xs):\n    return sum(xs)\n",
        steps=["xs = [1, 2]", "total(xs)", "xs.append(3)", "total(xs)"],
    ),
]
```

Runs are numbered 0 to 4: Run All three times, then the restart, then
twice more. A shape can give its own `plan`: a list of `"all"`,
`"restart"` and cell indexes to run one cell alone. For example,
`plan=["all", 1, "restart", "all", 1]` re-runs cell 1 after a Run All.
Edits are then keyed by the plan's runs, with restarts not counted. A
custom plan has no REBUILT verdict. Leave out `%cash_*` magics and `import cash`, because the
checker sets cash up itself. Keep cells deterministic: no unseeded random
numbers, clock readings or addresses in the output. A cell that is slower
than the 10 ms floor is cached in every config. A cheaper cell is cached
only under `floors0`.

`shapes/starter.py` holds repros taken from earlier reviews. The
`*_corrected.py` files hold shapes rewritten after a triage found the first
adaptation unfaithful. `shapes/smoke.py` holds two quick shapes that check
the checker itself.

## Limits

- It runs one kernel at a time per shape, so a shape takes about 30 s.
  Use `-j` to run several shapes at once.
- Cells must keep their count across edits.
- It compares values, not cash's warnings. A wrong warning label shows only
  in the statuses and stderr, not as a mismatch.

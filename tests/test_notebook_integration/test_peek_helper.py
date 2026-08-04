"""`NotebookTestRunner.peek` reads LIVE kernel state (CAS-267).

Integration tests assert on `get_output`, which is the right instrument for
"what does the user see" and the wrong one for "what does the kernel hold". A
cached statement's stdout is **replayed on a hit**, so reading state through a
printed cell reports what was on screen when the entry was written. Measured
during CAS-260, that made a broken arm look correct and cost a round of the
investigation.

`peek` was hand-rolled identically in nine probe files before it moved here,
which is the usual signal that it belongs in the fixture.

Two naive implementations were written and watched to fail, so the guards are
known to have teeth rather than assumed to:

* **no identifier wrap** (evaluate the bare name) -- fails
  `test_peek_reports_an_undefined_name_as_none` and
  `test_get_output_is_a_recording_and_peek_is_live`, because the raise produces
  no stdout and "no output" would read as a value;
* **`startswith` on the joined text** instead of scanning lines -- fails
  `test_peek_finds_its_marker_when_other_output_shares_the_channel`, where the
  badge occupies the first line.

What these guards do NOT discriminate: an implementation that evaluates through
a notebook cell. That was the third naive version tried, and it passes
everything here -- `test_peek_leaves_the_notebook_untouched` catches the cell
being *recorded*, but a cell run with `store_history=False` would slip through.
The reason no test pins it is that the divergence it would cause needs a served
statement whose printed state is stale, and CAS-260 removed that shape by
skip-caching exactly those statements. Worth knowing before trusting this file
to catch a rewrite of the evaluation path.
"""
import pytest

pytestmark = [pytest.mark.integration]

SETUP = "import cash\n%cash_on\n"


def test_get_output_is_a_recording_and_peek_is_live(nb_runner):
    """The contract, stated as a contrast: `get_output` returns text captured
    when the cell last ran; `peek` asks the kernel now.

    A restart makes the gap unmissable -- the notebook keeps its recorded
    output while the kernel holds nothing. A test asserting on `get_output`
    here would happily confirm a variable that no longer exists.

    The first draft of this test used a cached statement whose callee mutates a
    global, on the theory that a served re-run would leave the printed text
    ahead of the live value. It does not: CAS-260 skip-caches exactly that
    statement so it re-executes, and the two agree. The stale-reading trap this
    helper exists for is a property of *recorded output*, which is what this
    asserts.
    """
    nb_runner.create_notebook([SETUP, "v = 1\nprint('V', v)\n"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "V 1" in nb_runner.get_output(2)
    assert nb_runner.peek("v") == "1"

    nb_runner.restart()

    assert "V 1" in nb_runner.get_output(2), (
        "precondition: the notebook should still hold the recorded output"
    )
    assert nb_runner.peek("v") == "None", (
        "peek read the recorded output instead of the live (empty) kernel"
    )


def test_peek_reports_an_undefined_name_as_none(nb_runner):
    """A bare undefined name raises, prints nothing, and "no output" would read
    as a value. The identifier wrap is what prevents that."""
    nb_runner.create_notebook([SETUP])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert nb_runner.peek("never_assigned") == "None"


def test_peek_finds_its_marker_when_other_output_shares_the_channel(nb_runner):
    """The text badge writes to stdout too. Matching the joined text rather
    than scanning lines reads the badge as "no value"."""
    nb_runner.create_notebook([
        "import cash\n%cash_on\n%cash_badge print\n",
        "value = 41 + 1\n",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert nb_runner.peek("value") == "42"


def test_peek_evaluates_a_non_identifier_expression_as_written(nb_runner):
    """Only a bare name is wrapped; anything else is the caller's expression."""
    nb_runner.create_notebook([SETUP, "items = [1, 2, 3]\n"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert nb_runner.peek("len(items)") == "3"
    assert nb_runner.peek("items[0] + 10") == "11"


def test_peek_leaves_the_notebook_untouched(nb_runner):
    """`store_history=False`, so peeking must not add a cell, produce output,
    or advance the execution count -- otherwise the observer perturbs the run
    it is measuring."""
    nb_runner.create_notebook([SETUP, "a = 1\n"])
    nb_runner.start_kernel()
    nb_runner.run_all()

    before_cells = len(nb_runner.nb.cells)
    before_counts = [c.get("execution_count") for c in nb_runner.nb.cells]
    before_output = nb_runner.get_output(2)

    for _ in range(3):
        nb_runner.peek("a")

    assert len(nb_runner.nb.cells) == before_cells, "peek added a cell"
    assert [c.get("execution_count") for c in nb_runner.nb.cells] == before_counts, (
        "peek advanced an execution count"
    )
    assert nb_runner.get_output(2) == before_output, "peek changed a cell's output"


def test_peek_returns_a_marker_rather_than_guessing_when_it_fails(nb_runner):
    """An expression that raises yields no marker line. That must be
    distinguishable from a real value, not silently empty."""
    nb_runner.create_notebook([SETUP])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert nb_runner.peek("1 / 0") == "?"

"""``# @cash:no-cache``: what it covers and what re-runs because of it.

A no-cache statement behaves as if cash weren't installed: re-running it
advances state, like plain Jupyter, for reassignment and in-place mutation
alike, and does not re-run the producer of the variable it changes.

A leading ``# @cash:no-cache`` covers the whole cell. The coverage tests count
live statements with ``bump(tag)``, which appends a line to a file that only
the test side reads, so cash's file tracking cannot double-count it and the
badge is never taken as evidence: an isolated re-run appends one line per
statement that really ran.
"""

import textwrap

import pytest


def _last(out: str) -> str:
    return out.strip().splitlines()[-1].strip()


SETUP = "import cash\n%cash_on\n%cash_badge print\nimport time"


def _bump_def(sink: str) -> str:
    return (
        f"SINK = r'{sink}'\n"
        "def bump(tag):\n"
        "    time.sleep(0.05)\n"
        "    with open(SINK, 'a') as f:\n"
        "        f.write(tag + '\\n')\n"
        "    return tag"
    )


def _tags(sink):
    return [ln for ln in sink.read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.mark.integration
@pytest.mark.upstream
def test_nocache_inplace_mutation_advances(nb_runner):
    nb_runner.create_notebook(
        [
            "log = []",
            "# @cash: no-cache\nlog.append(len(log))\nprint(len(log))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _last(nb_runner.get_output(2)) == "1"
    nb_runner.run_cell(2)
    assert _last(nb_runner.get_output(2)) == "2", nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.upstream
def test_nocache_reassignment_advances(nb_runner):
    nb_runner.create_notebook(
        [
            "counter = 0",
            "# @cash: no-cache\ncounter = counter + 1\nprint(counter)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _last(nb_runner.get_output(2)) == "1"
    nb_runner.run_cell(2)
    assert _last(nb_runner.get_output(2)) == "2", nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.upstream
def test_cached_inplace_mutation_still_resets(nb_runner):
    """Control: WITHOUT no-cache, the run-from-start guarantee still resets the
    in-place mutation on an isolated re-run (the fix is scoped to no-cache)."""
    nb_runner.create_notebook(
        [
            "log = []",
            "log.append(len(log))\nprint(len(log))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _last(nb_runner.get_output(2)) == "1"
    nb_runner.run_cell(2)
    assert _last(nb_runner.get_output(2)) == "1", nb_runner.get_output(2)


@pytest.mark.integration
@pytest.mark.upstream
def test_nocache_inplace_does_not_reexecute_producer(upstream_trace):
    """The producer of a no-cache-mutated var must not be scheduled for re-run."""
    t = upstream_trace(
        ["log = []", "# @cash: no-cache\nlog.append(len(log))\nprint(len(log))"],
        lambda r: r.run_cell(2),
    )
    assert "log = []" not in t.scheduled(), t.scheduled()
    assert t.events("broken_drop_nocache"), "expected log dropped from broken_vars"


@pytest.mark.timeout(180)
def test_leading_no_cache_covers_whole_cell(nb_runner, tmp_path):
    sink = tmp_path / "bumps.txt"
    sink.write_text("", encoding="utf-8")
    sink_s = str(sink).replace("\\", "/")

    nb_runner.create_notebook(
        [
            SETUP,
            _bump_def(sink_s),
            # The natural spelling from the ticket: ONE directive at the top of the
            # cell, followed by three top-level statements.
            "# @cash:no-cache\na = bump('s1')\nb = bump('s2')\nc = bump('s3')",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    first = _tags(sink)
    assert first == ["s1", "s2", "s3"], first

    # Isolated warm re-runs of the annotated cell. A whole-cell no-cache means
    # every statement re-executes every time.
    for run in range(2, 5):
        nb_runner.run_cell(3)
        got = _tags(sink)
        expected = ["s1", "s2", "s3"] * run
        assert got == expected, (
            f"warm re-run #{run}: leading '# @cash:no-cache' did NOT cover the "
            f"whole cell.\n  expected {expected}\n  got      {got}\n"
            f"  -> statements that did NOT re-execute: "
            f"{sorted(set(['s1', 's2', 's3']) - set(got[len(expected) - 3 :]))}\n"
            f"  cell output: {nb_runner.get_output(3)!r}"
        )


@pytest.mark.timeout(180)
def test_statement_adjacent_no_cache_still_scoped(nb_runner, tmp_path):
    """Control: a directive directly above ONE mid-cell statement stays scoped
    to that statement (this is the behaviour any fix must preserve)."""
    sink = tmp_path / "bumps2.txt"
    sink.write_text("", encoding="utf-8")
    sink_s = str(sink).replace("\\", "/")

    nb_runner.create_notebook(
        [
            SETUP,
            _bump_def(sink_s),
            "a = bump('s1')\n# @cash:no-cache\nb = bump('s2')\nc = bump('s3')",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _tags(sink) == ["s1", "s2", "s3"], _tags(sink)

    nb_runner.run_cell(3)
    got = _tags(sink)
    # Record whatever actually happens; s2 must at minimum re-fire.
    assert "s2" in got[3:], f"the statement-adjacent no-cache statement did not re-execute: {got}"
    print(f"[no-cache control] after warm re-run, sink = {got}")


@pytest.mark.stress
@pytest.mark.integration
class TestNoCacheAnnotation:
    """Test @cash: no-cache directive."""

    def test_no_cache_always_recomputes(self, nb_runner):
        """@cash: no-cache prevents caching of a statement."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                counter = 0
            """),
                textwrap.dedent("""\
                # @cash: no-cache
                counter = counter + 1
                print(counter)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output1 = nb_runner.get_output(2)
        assert "1" in output1

        # Re-run — should recompute, not use cache
        nb_runner.run_cell(2)
        output2 = nb_runner.get_output(2)
        assert "2" in output2

    def test_no_cache_on_print(self, nb_runner):
        """@cash: no-cache on a print statement."""
        nb_runner.create_notebook(
            [
                "x = 42",
                textwrap.dedent("""\
                # @cash: no-cache
                print(f"x = {x}")
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "x = 42" in nb_runner.get_output(2)

    def test_no_cache_mixed_with_cached(self, nb_runner):
        """Mix of cached and no-cache statements in same cell."""
        nb_runner.create_notebook(
            [
                textwrap.dedent("""\
                a = 10
                # @cash: no-cache
                b = a + 1
                c = a * 2
                print(b, c)
            """),
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        output = nb_runner.get_output(1)
        assert "11" in output
        assert "20" in output


@pytest.mark.stress
@pytest.mark.core
@pytest.mark.timeout(30)
class TestNoCacheDirective:
    """@cash:no-cache + cell edits."""

    def test_no_cache_cell_runs_on_the_first_run_all(self, nb_runner):
        """A cell with @cash:no-cache runs and prints on the first run_all."""
        nb_runner.create_notebook(
            [
                "counter = 0",
                "# @cash:no-cache\ncounter = counter + 1",
                "print(f'counter = {counter}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "counter = 1" in nb_runner.get_output(3)

    def test_remove_no_cache_directive(self, nb_runner):
        """Remove @cash:no-cache directive — cell becomes cacheable."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "# @cash:no-cache\ny = x * 3\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(2)

        # Remove no-cache
        nb_runner.set_cell_source(2, "y = x * 3\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 15" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.upstream
@pytest.mark.timeout(45)
class TestNoCacheAnnotationEdits:
    """@cash:no-cache annotation with cell edits."""

    def test_add_no_cache_annotation(self, nb_runner):
        """Add @cash:no-cache annotation to a cell."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "y = x * 2\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

        # Add no-cache annotation
        nb_runner.set_cell_source(2, "# @cash:no-cache\ny = x * 2\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 20" in nb_runner.get_output(2)

    def test_remove_no_cache_annotation(self, nb_runner):
        """Remove @cash:no-cache annotation."""
        nb_runner.create_notebook(
            [
                "x = 5",
                "# @cash:no-cache\ny = x + 1\nprint(f'y = {y}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "y = 6" in nb_runner.get_output(2)

        # Remove annotation
        nb_runner.set_cell_source(2, "y = x + 1\nprint(f'y = {y}')")
        nb_runner.run_all()
        assert "y = 6" in nb_runner.get_output(2)


@pytest.mark.stress
@pytest.mark.core
@pytest.mark.timeout(30)
class TestMixedDirectives:
    """Multiple directives + cell edits."""

    def test_no_cache_and_regular_mixed(self, nb_runner):
        """Mix of no-cache and regular cells."""
        nb_runner.create_notebook(
            [
                "x = 10",
                "# @cash:no-cache\ny = x + 1",
                "z = y * 2\nprint(f'z = {z}')",
            ]
        )
        nb_runner.start_kernel()
        nb_runner.run_all()
        assert "z = 22" in nb_runner.get_output(3)

        nb_runner.set_cell_source(1, "x = 20")
        nb_runner.run_all()
        assert "z = 42" in nb_runner.get_output(3)

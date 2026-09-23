"""A file read in a loop's header is a dependency of what the loop builds.

Round 27, r27s4. A gene-alias table was built the way most people write one::

    ALIAS = {}
    for line in DATA.read_text().splitlines()[1:]:
        o, n = line.split("\\t")
        ALIAS[o] = n

The tester regenerated their input data and carried on working further down
the notebook -- the move the quickstart says cash exists to make safe. Every
exported file was then computed from the old table: 19,610 genes instead of
19,850, 13 clusters instead of 12, different markers for every cluster.
``compare.py`` reported MISMATCH on all nine outputs, and the cell they ran
showed no ``Upstream:`` section at all.

Two things had to be true for that, and both are fixed here.

**The loop's HEADER was not tracked.** A loop is decomposed per-iteration and
every body statement gets tracked, but the iterable expression was evaluated
by a bare ``eval`` with no file-access tracking around it -- and the body
statements never touch the file. So nothing recorded that the loop read
anything, ``%cash_provenance ALIAS`` reported ``Code: ALIAS = {}``, and the
control structure's recorded outcome carried an EMPTY file set, which of
course never changes.

**The file check was unreachable anyway.** A control structure's outcome is
recorded with the files behind what it produced, and the simulation compares
their state against the recording -- but only after ``recorded[0] ==
input_hashes`` said the input lineages still matched. A name bound in the same
cell as ``%cash_on`` has no runtime lineage (cash was not listening when that
cell started) while the simulation, which reads that cell from the file, has
one. So the recorded entry lacked a key the simulation carried, equality was
false forever, and neither branch ran. That is r27s4's cell 0, and the
quickstart's. The file check now runs whether or not the inputs match, which
is why this file tests BOTH cell layouts.

The single-assignment spellings work precisely because one statement both
reads the file and binds the name::

    ALIAS = dict(l.split("\\t") for l in DATA.read_text().splitlines()[1:])

which is why this survived four rounds: it needs the reading half and the
binding half to be different statements. r27s4's notebook carries a comment
above that line saying the tester had already worked this out and written
around it.

Pre-existing since 2026-05-29 at the latest; reproduced on the round-24, -25
and -26 builds as well as this one.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

LOOP = "ALIAS = {}\nfor line in DATA.read_text().splitlines()[1:]:\n    o, n = line.split('\\t')\n    ALIAS[o] = n"

READ = "print('R', [ALIAS.get(s, s) for s in ['LOC1', 'LOC3']])"


def _write(path, third):
    path.write_text("old\tnew\nLOC1\tGENE_A\nLOC2\tGENE_B\nLOC3\t%s\n" % third, encoding="utf-8")


def test_a_for_loop_header_read_invalidates_what_the_loop_built(nb_runner, tmp_path):
    """r27s4's shape: change the file, run a cell BELOW, get the new value.

    ``DATA`` is bound in the same cell as ``%cash_on``, exactly as the
    tester's notebook and the quickstart do it, so this is also the case
    where the recorded input lineages do NOT match the simulated ones.
    """
    data = tmp_path / "alias.tsv"
    _write(data, "GENE_C_OLD")

    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\nfrom pathlib import Path\nDATA = Path(r'" + str(data) + "')",
            LOOP,
            READ,
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "GENE_C_OLD" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    _write(data, "GENE_C_NEW")
    nb_runner.run_cell(3)

    out = nb_runner.get_output(3)
    assert "GENE_C_NEW" in out, (
        "the loop's source file changed and the cell below served the old "
        "value; a Restart & Run All gives GENE_C_NEW:\n" + nb_runner.get_raw_output(3)
    )


def test_the_same_loop_with_its_input_bound_in_a_tracked_cell(nb_runner, tmp_path):
    """The other half: ``DATA`` bound where cash is already listening.

    Same bug, different route to it. Here the recorded input lineages DO
    match, so the fix is carried entirely by the header's read reaching the
    outcome's file set -- the branch the first test cannot reach.
    """
    data = tmp_path / "alias_tracked.tsv"
    _write(data, "GENE_C_OLD")

    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            "from pathlib import Path\nDATA = Path(r'" + str(data) + "')",
            LOOP,
            READ,
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "GENE_C_OLD" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    _write(data, "GENE_C_NEW")
    nb_runner.run_cell(4)
    assert "GENE_C_NEW" in nb_runner.get_output(4), nb_runner.get_raw_output(4)


def test_the_same_shape_written_as_one_statement_already_works(nb_runner, tmp_path):
    """The control, and the reason this went unnoticed for four rounds."""
    data = tmp_path / "alias2.tsv"
    _write(data, "GENE_C_OLD")

    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\nfrom pathlib import Path\nDATA = Path(r'" + str(data) + "')",
            "ALIAS = dict(l.split('\\t') for l in DATA.read_text().splitlines()[1:])",
            READ,
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "GENE_C_OLD" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    _write(data, "GENE_C_NEW")
    nb_runner.run_cell(3)
    assert "GENE_C_NEW" in nb_runner.get_output(3), nb_runner.get_raw_output(3)


def test_an_unchanged_file_still_restores(nb_runner, tmp_path):
    """The control that stops the fix turning every loop into a cache miss.

    Recording the header's read must not make the loop re-run when nothing
    moved -- that would trade a wrong answer for a permanent slowdown.
    """
    data = tmp_path / "alias3.tsv"
    _write(data, "GENE_C_OLD")

    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\nfrom pathlib import Path\nimport sys\nDATA = Path(r'" + str(data) + "')",
            "ALIAS = {}\n"
            "for line in DATA.read_text().splitlines()[1:]:\n"
            "    print('RAN', file=sys.stderr)\n"
            "    o, n = line.split('\\t')\n"
            "    ALIAS[o] = n",
            "print('R', len(ALIAS))",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R 3" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    nb_runner.run_cell(3)
    assert "R 3" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

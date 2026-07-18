"""CAS-174 verification: does %cash_on echo EVERY statement's repr?

The claim: with cash on, every bare expression statement in a cell has its repr
echoed, where plain IPython echoes only the cell's LAST expression. Ground truth
is the literal cell output text under %cash_on vs %cash_off in ONE real kernel,
so kernel/IPython version is held constant across the comparison.

Distinct variable names (``a``/``b``) keep the two cells' sources distinct so
they cannot collide on a cache key; the echo behaviour under test does not
depend on the identifier.
"""
import pytest

pytestmark = [pytest.mark.timeout(120)]


def test_cash_on_repr_echo_matches_cash_off(nb_runner):
    nb_runner.create_notebook([
        "import cash\n%cash_on",
        "a = 1\na + 1\na + 2\na + 3",      # cash ON
        "%cash_off",
        "b = 1\nb + 1\nb + 2\nb + 3",      # cash OFF
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    on = nb_runner.get_output(2).strip()
    off = nb_runner.get_output(4).strip()

    print(f"\ncash ON  output: {on!r}")
    print(f"cash OFF output: {off!r}")

    # Plain IPython echoes only the final expression -> "4".
    assert off == "4", f"baseline (cash off) is not plain-IPython semantics: {off!r}"
    assert on == off, (
        f"CAS-174 reproduces: cash on echoed {on!r}, cash off echoed {off!r}"
    )


def test_trailing_semicolon_suppresses_echo_under_cash(nb_runner):
    """CAS-96 adjacent: a trailing ';' must suppress the final echo, with cash on."""
    nb_runner.create_notebook([
        "import cash\n%cash_on",
        "a = 1\na + 1\na + 3;",           # cash ON, suppressed
        "%cash_off",
        "b = 1\nb + 1\nb + 3;",           # cash OFF, suppressed
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    on = nb_runner.get_output(2).strip()
    off = nb_runner.get_output(4).strip()

    print(f"\ncash ON  (semicolon) output: {on!r}")
    print(f"cash OFF (semicolon) output: {off!r}")

    assert off == "", f"baseline (cash off) did not suppress on ';': {off!r}"
    assert on == off, (
        f"trailing ';' suppression differs under cash: on={on!r} off={off!r}"
    )


def test_mixed_print_and_expressions(nb_runner):
    """The ticket's real-world shape: real work interleaved with expressions."""
    nb_runner.create_notebook([
        "import cash\n%cash_on",
        "a = [1, 2, 3]\nlen(a)\nsum(a)\nprint('work')\nsorted(a)",
        "%cash_off",
        "b = [1, 2, 3]\nlen(b)\nsum(b)\nprint('work')\nsorted(b)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()

    on = nb_runner.get_output(2).strip()
    off = nb_runner.get_output(4).strip()

    print(f"\ncash ON  (mixed) output: {on!r}")
    print(f"cash OFF (mixed) output: {off!r}")

    assert on == off, f"CAS-174 reproduces (mixed cell): on={on!r} off={off!r}"

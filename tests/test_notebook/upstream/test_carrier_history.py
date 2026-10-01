"""The drawing history that stands in for a figure's lineage in write provenance.

Runtime and simulation each compute it from their own statements, so what
must hold is: same drawing -> same fingerprint whatever the figure's own
lineage, and anything that can change the picture -> a different one or none.
"""

from cash.notebook.carrier_history import carrier_history_fingerprint

CHART = [
    ("data = load()", {"load": "L1"}),
    ("fig, ax = plt.subplots()", {"plt": "P"}),
    ("ax.bar(range(len(data)), data)", {"ax": "A1", "data": "D1"}),
    ("ax.set_title('first')", {"ax": "A2"}),
    ("fig.tight_layout()", {"fig": "F1"}),
]


def _with(statements, k, code=None, lineages=None):
    out = list(statements)
    old_code, old_lineages = out[k]
    out[k] = (code if code is not None else old_code, lineages if lineages is not None else old_lineages)
    return out


def test_the_figure_s_own_lineage_does_not_count():
    """What a restarted kernel cannot reproduce must not decide it."""
    other = _with(_with(CHART, 2, lineages={"ax": "A9", "data": "D1"}), 4, lineages={"fig": "F9"})
    assert carrier_history_fingerprint(CHART, "fig") == carrier_history_fingerprint(other, "fig")
    assert carrier_history_fingerprint(CHART, "fig") is not None


def test_the_data_a_fill_reads_counts():
    changed = _with(CHART, 2, lineages={"ax": "A1", "data": "D2"})
    assert carrier_history_fingerprint(changed, "fig") != carrier_history_fingerprint(CHART, "fig")


def test_an_edited_fill_counts():
    changed = _with(CHART, 3, code="ax.set_title('renamed')")
    assert carrier_history_fingerprint(changed, "fig") != carrier_history_fingerprint(CHART, "fig")


def test_a_statement_not_recognised_as_drawing_still_counts():
    """``ax2 = axes[1]`` draws nothing and ``ax2`` is not what ``subplots``
    bound -- but the bar after it is part of the picture."""
    chart = [
        ("fig, axes = plt.subplots(1, 2)", {"plt": "P"}),
        ("ax2 = axes[1]", {"axes": "X1"}),
        ("ax2.bar(range(3), totals)", {"ax2": "B1", "totals": "T1"}),
    ]
    changed = _with(chart, 2, lineages={"ax2": "B1", "totals": "T2"})
    assert carrier_history_fingerprint(changed, "fig") != carrier_history_fingerprint(chart, "fig")


def test_what_ran_before_the_figure_was_made_does_not_count():
    changed = _with(CHART, 0, lineages={"load": "L2"})
    assert carrier_history_fingerprint(changed, "fig") == carrier_history_fingerprint(CHART, "fig")


def test_each_figure_has_its_own_history():
    """``fig`` is bound again for a second chart: the first chart's bars are
    not part of it, the second chart's are."""
    second = CHART + [
        ("plt.close(fig)", {"plt": "P", "fig": "F2"}),
        ("fig, axes = plt.subplots(1, 2)", {"plt": "P"}),
        ("axes[0].bar(range(3), data[:3])", {"axes": "X1", "data": "D1"}),
    ]
    first_bars = _with(second, 2, lineages={"ax": "A1", "data": "D2"})
    second_bars = _with(second, 7, lineages={"axes": "X1", "data": "D2"})
    assert carrier_history_fingerprint(first_bars, "fig") == carrier_history_fingerprint(second, "fig")
    assert carrier_history_fingerprint(second_bars, "fig") != carrier_history_fingerprint(second, "fig")


def test_statements_without_an_entry_on_either_side_are_left_out():
    """``del`` and a bare name get no simulation trace entry."""
    padded = CHART[:3] + [("del tmp", {"tmp": "T"}), ("ax", {"ax": "A1"})] + CHART[3:]
    assert carrier_history_fingerprint(padded, "fig") == carrier_history_fingerprint(CHART, "fig")


def test_no_fingerprint_for_a_figure_not_made_here():
    assert carrier_history_fingerprint([("fig = plt.gcf()", {"plt": "P"})], "fig") is None
    assert carrier_history_fingerprint(CHART, "other") is None


def test_a_loop_in_the_span_is_one_statement_of_the_history():
    """A loop is one trace entry to the simulation, and the runtime logs it the
    same way (`StatementProcessor.begin_control_log`). It used to make the
    history unknowable -- and a chart drawn through ``for ax in axes`` was then
    never known to be current, or to be stale."""
    looped = CHART + [("for c in cols:\n    ax.plot(df[c])", {"cols": "C", "ax": "A2", "df": "D"})]
    fingerprint = carrier_history_fingerprint(looped, "fig")
    assert fingerprint is not None
    redrawn = CHART + [("for c in cols:\n    ax.plot(df[c])", {"cols": "C", "ax": "A2", "df": "D2"})]
    assert carrier_history_fingerprint(redrawn, "fig") != fingerprint


def test_no_fingerprint_when_the_span_holds_a_branch():
    branched = CHART + [("if flag:\n    ax.plot(df['a'])", {"flag": "B", "ax": "A2", "df": "D"})]
    assert carrier_history_fingerprint(branched, "fig") is None


def test_no_fingerprint_when_the_span_reads_a_file():
    """A lineage does not show what ``imread`` found on disk."""
    logo = CHART + [("ax.imshow(plt.imread('logo.png'))", {"ax": "A2", "plt": "P"})]
    assert carrier_history_fingerprint(logo, "fig") is None

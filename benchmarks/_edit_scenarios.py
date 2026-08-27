"""Edit scenarios for the notebook edit benchmark.

The overhead benchmark measures cash on a notebook that has not changed:
off / cold / warm-session / warm-restart all run the *same* cells. That is
the best case, and it is not the case anyone actually works in. The case
that matters is the iteration loop -- change one cell, run again -- which is
where every open cache-effectiveness ticket lives (#5, #51, #62): "an
unrelated upstream edit re-executes the loop chain".

This module builds the edits and attributes the cost.

The measurement question
------------------------
After an edit, which statements did cash recompute that it *could have*
restored? Not "which statements computed" -- plenty of statements are
uncacheable, or below the cost-model threshold, and recomputing those is
correct behaviour, not waste.

So waste is defined against a **no-edit control run**:

    waste = statements that RESTORE when nothing is edited,
            but COMPUTE after an edit that cannot have invalidated them.

A statement that computes in the control too is not counted. That baseline
is the whole reason this module is more than a diff of two timings.

Edit kinds
----------
``comment``      Append a comment line. Nothing at all should recompute --
                 cash hashes statements comment-insensitively, so this is a
                 direct test of that claim on a real notebook.

``null-assign``  Append ``_cash_bench_probe_N = N``, a fresh name nothing
                 reads. The new statement must run. Nothing else may.

``linked``       The positive control, and the reason to trust the other
                 two. Appends a probe at the site AND a *reader* of that
                 probe in the last cell, then changes the probe's value.
                 The reader MUST recompute. If it does not, this harness
                 cannot see a real dependency and every zero-waste result
                 it reports is meaningless.

Without ``linked``, a harness that silently failed to detect any
recomputation would score a perfect zero on every scenario.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from benchmarks._overhead_io import CodeCell

PROBE_PREFIX = "_cash_bench_probe_"
SINK_PREFIX = "_cash_bench_sink_"

# Kinds whose expected downstream closure is empty: after the edit, every
# statement outside the edited cell should still restore.
NULL_KINDS = ("comment", "null-assign")


@dataclass(frozen=True)
class EditScenario:
    """One edit applied to one cell, plus what it is expected to cost.

    Attributes:
        kind: One of ``comment`` / ``null-assign`` / ``linked``.
        site: Index among *code* cells of the edited cell.
        label: Stable identifier used in results and logs.

    The two halves of a scenario -- what to run when priming the cache and
    what to run when measuring -- come from :func:`build_cells`, not from a
    field here, because for ``linked`` the priming half already carries the
    probe/sink pair at its "before" value: a control that appeared for the
    first time in the measured run would recompute simply for being new
    code, and would prove nothing.
    """
    kind: str
    site: int
    label: str

    def is_null(self) -> bool:
        """True when nothing outside the edited cell should recompute."""
        return self.kind in NULL_KINDS


@dataclass
class SiteRanking:
    """A candidate edit site and the downstream compute it puts at risk."""
    site: int
    downstream_restorable_seconds: float
    downstream_restorable_count: int


def _append_line(cell: CodeCell, line: str) -> CodeCell:
    """Return ``cell`` with ``line`` appended as a new final line.

    Appending rather than inserting is deliberate: a cell may open with a
    cell magic (which must stay on line 1) or with a multi-line statement,
    so there is no other position that is safe on every cell without
    parsing. The sharper test -- a comment *between* two statements -- needs
    that parsing and is not attempted here.
    """
    source = cell.source
    if source and not source.endswith("\n"):
        source += "\n"
    return replace(cell, source=source + line + "\n")


def rank_sites(
    cells: list[CodeCell],
    restorable_seconds_by_cell: dict[int, float],
    restorable_count_by_cell: dict[int, int],
) -> list[SiteRanking]:
    """Rank each code cell by how much restorable compute sits *below* it.

    Editing the last cell of a notebook proves nothing -- there is nothing
    downstream to be wrongly invalidated. Editing the cell above an
    expensive chain proves a great deal. Ranking by downstream restorable
    time picks the sites where over-invalidation would actually cost the
    user something, rather than spreading edits blindly across the file.
    """
    ranked: list[SiteRanking] = []
    for cell in cells:
        below = [c.index for c in cells if c.index > cell.index]
        ranked.append(SiteRanking(
            site=cell.index,
            downstream_restorable_seconds=sum(
                restorable_seconds_by_cell.get(i, 0.0) for i in below),
            downstream_restorable_count=sum(
                restorable_count_by_cell.get(i, 0) for i in below),
        ))
    return ranked


def plan_scenarios(
    cells: list[CodeCell],
    restorable_seconds_by_cell: dict[int, float],
    restorable_count_by_cell: dict[int, int],
    max_sites: int = 3,
) -> list[EditScenario]:
    """Choose edit sites and build the scenario list.

    Sites are chosen from the control run's data, not from fixed positions:
    a cell is only a useful site if something restorable sits below it. To
    keep the sites spread across the notebook rather than clustered on one
    hot chain, the candidates are split into ``max_sites`` contiguous bands
    and the best candidate in each band is taken.

    One ``linked`` control is emitted, at the middle site. One is enough to
    answer "can this harness see a real dependency"; more would cost a prime
    run each for no extra signal.
    """
    ranked = [r for r in rank_sites(
        cells, restorable_seconds_by_cell, restorable_count_by_cell)
        if r.downstream_restorable_count > 0]
    if not ranked:
        return []

    band_size = max(1, len(ranked) // max_sites)
    sites: list[int] = []
    for band_start in range(0, len(ranked), band_size):
        band = ranked[band_start:band_start + band_size]
        if not band:
            continue
        best = max(band, key=lambda r: (r.downstream_restorable_seconds,
                                        r.downstream_restorable_count))
        sites.append(best.site)
        if len(sites) == max_sites:
            break

    scenarios: list[EditScenario] = []
    for site in sites:
        for kind in NULL_KINDS:
            scenarios.append(EditScenario(
                kind=kind, site=site, label=f"{kind}@cell{site}"))
    if sites:
        control_site = sites[len(sites) // 2]
        scenarios.append(EditScenario(
            kind="linked", site=control_site,
            label=f"linked@cell{control_site}"))
    return scenarios


def probe_names(scenario: EditScenario) -> tuple[str, str]:
    """The probe and sink names this scenario injects, if any."""
    return (f"{PROBE_PREFIX}{scenario.site}", f"{SINK_PREFIX}{scenario.site}")


def build_cells(
    cells: list[CodeCell],
    scenario: EditScenario,
    *,
    edited: bool,
) -> list[CodeCell]:
    """Return the cell list for one half of a scenario.

    ``edited=False`` builds the priming run, ``edited=True`` the measured
    one. For the null kinds the priming run is the notebook untouched. For
    ``linked`` both runs carry the probe/sink pair and only the probe's
    *value* differs, so the sink's recomputation is caused by the value
    change and not by the code being new.
    """
    probe, sink = probe_names(scenario)
    out = list(cells)

    if scenario.kind == "comment":
        if not edited:
            return out
        return [_append_line(c, f"# cash-bench edit probe {scenario.site}")
                if c.index == scenario.site else c for c in out]

    if scenario.kind == "null-assign":
        if not edited:
            return out
        return [_append_line(c, f"{probe} = {scenario.site}")
                if c.index == scenario.site else c for c in out]

    if scenario.kind == "linked":
        value = scenario.site + (1000 if edited else 0)
        last = max(c.index for c in out)
        rebuilt: list[CodeCell] = []
        for c in out:
            if c.index == scenario.site:
                c = _append_line(c, f"{probe} = {value}")
            if c.index == last:
                c = _append_line(c, f"{sink} = {probe} + 1")
            rebuilt.append(c)
        return rebuilt

    raise ValueError(f"unknown edit kind: {scenario.kind!r}")


@dataclass
class WastedStatement:
    """A statement that restored in the control but recomputed after an edit."""
    cell_index: int
    code: str
    execution_seconds: float
    uncacheable_reasons: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    label: str
    kind: str
    site: int
    wasted_seconds: float
    wasted_count: int
    restorable_seconds: float
    restorable_count: int
    wasted: list[WastedStatement] = field(default_factory=list)
    # `linked` only: did the injected reader recompute? None for other kinds.
    control_sink_recomputed: bool | None = None

    @property
    def waste_fraction(self) -> float:
        if self.restorable_seconds <= 0:
            return 0.0
        return self.wasted_seconds / self.restorable_seconds


def _key_counter(metrics_by_cell: dict[int, list]) -> dict[tuple[int, str], list]:
    """Index statements by (cell, code), preserving duplicates positionally.

    A cell can contain the same source line twice. Matching on code alone
    would pair the first occurrence in one run with the second in another
    and mis-attribute both.
    """
    out: dict[tuple[int, str], list] = {}
    for cell_index, metrics in metrics_by_cell.items():
        for m in metrics:
            out.setdefault((cell_index, m.code), []).append(m)
    return out


# A statement whose recompute costs less than this is not evidence of
# anything. cash declines to cache below a ~10ms floor, so near it the cost
# model's own decision flips on timing noise: the statement can be cached
# during the control run and refused during the edited one, which this
# accounting would read as "restored, then recomputed" -- waste, with no
# invalidation involved at all.
#
# That is not hypothetical. It turned the CI floor test red on macos-3.10
# and nowhere else: `base = np.arange(2_000_000)` costs 2.2ms on a fast box
# (never cached, never counted) and straddles the floor on a slow one. The
# margin is 2x the floor so a statement has to be unambiguously worth
# caching before its recomputation counts against anything.
MIN_WASTE_SECONDS = 0.020


def attribute_waste(
    scenario: EditScenario,
    control_by_cell: dict[int, list],
    edited_by_cell: dict[int, list],
    min_waste_seconds: float = MIN_WASTE_SECONDS,
) -> ScenarioResult:
    """Compare an edited run against the no-edit control.

    Only statements that (a) restored in the control, (b) live outside the
    edited cell, and (c) are not the harness's own injected probe/sink are
    eligible to count as waste. That triple filter is what keeps this from
    re-reporting the ordinary, correct recomputation of uncacheable work.
    """
    probe, sink = probe_names(scenario)
    control_index = _key_counter(control_by_cell)
    edited_index = _key_counter(edited_by_cell)

    wasted: list[WastedStatement] = []
    restorable_seconds = 0.0
    restorable_count = 0

    for (cell_index, code), control_metrics in control_index.items():
        if code.startswith(probe) or code.startswith(sink):
            continue
        edited_metrics = edited_index.get((cell_index, code), [])
        for position, control_metric in enumerate(control_metrics):
            if control_metric.status != "RESTORED":
                continue
            restorable_count += 1
            restorable_seconds += control_metric.execution_time or 0.0
            if cell_index == scenario.site:
                # The edited cell is expected to re-run its own statements
                # for null-assign; for `comment` it should not, but that is
                # reported separately rather than as downstream waste.
                continue
            if position >= len(edited_metrics):
                continue
            edited_metric = edited_metrics[position]
            if edited_metric.status != "RESTORED":
                if (edited_metric.execution_time or 0.0) < min_waste_seconds:
                    # Too cheap for the verdict to mean anything -- see
                    # MIN_WASTE_SECONDS. Counting it manufactures waste out of
                    # the cost model's own threshold noise.
                    continue
                wasted.append(WastedStatement(
                    cell_index=cell_index,
                    code=code[:160],
                    execution_seconds=edited_metric.execution_time or 0.0,
                    uncacheable_reasons=list(
                        getattr(edited_metric, "uncacheable_reasons", []) or []),
                ))

    sink_recomputed: bool | None = None
    if scenario.kind == "linked":
        sink_metrics = [m for metrics in edited_by_cell.values()
                        for m in metrics if m.code.startswith(sink)]
        sink_recomputed = any(m.status != "RESTORED" for m in sink_metrics)

    return ScenarioResult(
        label=scenario.label,
        kind=scenario.kind,
        site=scenario.site,
        wasted_seconds=sum(w.execution_seconds for w in wasted),
        wasted_count=len(wasted),
        restorable_seconds=restorable_seconds,
        restorable_count=restorable_count,
        wasted=wasted,
        control_sink_recomputed=sink_recomputed,
    )

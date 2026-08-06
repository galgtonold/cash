"""CAS-188: computed-RHS aliases (``b = a.attr`` / ``b = h[k]`` / ``b = f(a)``)
restore a stale COPY on a warm re-run — the half CAS-184 left open.

CAS-184 fixed the *bare* alias bind (``b = a``) by refusing to cache it. Every
other binding that also aliases a live mutable object is still cached, and on a
warm re-run cash rebinds the name to a DESERIALISED COPY taken before any later
mutation of the shared object. Identity with the source breaks and the mutation
becomes invisible through the alias.

Harness constraints inherited from CAS-184 — each one, left out, makes the probe
lie:

1. ``# @cash:persist`` on the bind. A pointer copy executes in ~0ms, under the
   10ms ``min_execution_time_to_cache_seconds`` floor, so cash normally never
   writes a cache entry and the bug stays LATENT.
2. Warm re-runs. Rep 0 is cold and correct; the divergence starts at rep 1.
3. ``# @cash:no-cache`` on EVERY probe print — the directive binds to ONE
   statement, so a partly-cached probe replays stale text. Each probe asserts
   ``RESTORED`` is absent so an invalid probe fails loudly instead of passing.

Each form gets its OWN notebook so attribution stays clean, because a SECOND and
distinct divergence shows up alongside this one: on the COLD run the mutation
cell's upstream simulation re-derives ``obj = Box()`` (badge: ``Upstream:
COMPUTED: obj = Box()``), swapping the live ``obj`` for a fresh instance while
the containers still hold the original. Measured on the subscript form:

    rep 0 (cold, bind EXECUTED): same False / same_holder True  / holder_is_obj False
    rep 1 (warm, bind RESTORED): same False / same_holder False / holder_is_obj True

Rep 0 is the upstream-simulation desync (``b`` still IS ``holder['k']``; it is
``obj`` that got replaced). Rep 1 is CAS-188 proper (the containers agree again,
but ``b`` is now a deserialised copy of neither). The failure messages below tag
each rep with the bind cell's state so the two never get conflated.
"""
import asyncio

import pytest

from tests.test_notebook_integration.conftest import NotebookTestRunner

pytestmark = [pytest.mark.upstream, pytest.mark.timeout(600)]

SETUP = "import cash\n%cash_on\n%cash_badge print"

WARM_REPS = 4  # 1 cold + 3 warm re-runs.

BUILD = (
    "class Box:\n"
    "    def __init__(self):\n"
    "        self.inner = []\n"
    "        self.tag = None\n"
    "obj = Box()\n"
    "holder = {'k': obj}\n"
    "lst = [obj]"
)

MUTATE = "obj.inner.append(42)\nobj.tag = 'mutated'"

# (id, bind statement, identity expression, value expression, expected value)
# CAS-206 blocks the container-mediated forms, and it is a DIFFERENT defect from
# CAS-188 (see the module docstring). The CAS-188 half — "the bind must not be
# restored from cache" — is asserted separately and unconditionally below, and
# passes for every form; these xfails cover only the end-to-end identity, which
# additionally needs CAS-206. Non-strict so they flip to XPASS the moment CAS-206
# lands rather than silently masking it.
_CAS206 = pytest.mark.xfail(
    reason="CAS-206: upstream re-derivation swaps live `obj` while the container "
           "keeps the original, so identity breaks on the COLD run too "
           "(bind cell reports 'executed', not 'RESTORED' — CAS-188's half is fixed)",
    strict=False,
)
# `b = list(lst)` aliases one level DOWN (`b[0] is lst[0]` while `b is not lst`).
# Refusing to cache the binding would not fix that, and `list(...)` is a call that
# can do real work, so it is deliberately outside the CAS-188 fix.
_ELEMENT_ALIAS = pytest.mark.xfail(
    reason="element-level aliasing through a freshly-built container; the binding "
           "itself is not the alias, so the CAS-188 refusal does not apply",
    strict=False,
)

FORMS = [
    ("attr", "b = obj.inner", "b is obj.inner", "b", "[42]"),
    pytest.param("subscript", "b = holder['k']", "b is obj",
                 "getattr(b, 'tag', 'MISSING')", "mutated", marks=_CAS206),
    pytest.param("index", "b = lst[0]", "b is obj",
                 "getattr(b, 'tag', 'MISSING')", "mutated", marks=_CAS206),
    pytest.param("call", "b = list(lst)", "b[0] is obj",
                 "getattr(b[0], 'tag', 'MISSING')", "mutated", marks=_ELEMENT_ALIAS),
    pytest.param("ternary", "b = obj if True else None", "b is obj",
                 "getattr(b, 'tag', 'MISSING')", "mutated", marks=_CAS206),
]

# Every form, unmarked: this is the CAS-188 property proper.
BIND_FORMS = [
    ("attr", "b = obj.inner"),
    ("subscript", "b = holder['k']"),
    ("index", "b = lst[0]"),
    ("ternary", "b = obj if True else None"),
]


@pytest.mark.parametrize("name,bind", BIND_FORMS)
def test_reference_bind_is_never_restored_from_cache(nb_runner, name, bind):
    """CAS-188 proper: a pure-dereference bind must re-execute, never restore.

    Restoring rebinds the target to a deserialised COPY, so the identity Python
    guarantees silently stops holding. A deref is free to re-run, so refusing to
    cache it is also strictly cheaper — both halves of the CAS-184 argument.

    This is asserted for every form and does NOT depend on CAS-206: it is a
    property of the bind statement alone.
    """
    nb_runner.create_notebook(_cells(bind, "b is obj", "getattr(b, 'tag', 'MISSING')"))
    nb_runner.start_kernel()
    nb_runner.assert_cash_active(True)
    nb_runner.run_all()

    for rep in range(3):
        nb_runner.run_cell(3)
        out = nb_runner.get_output(3)
        assert "RESTORED" not in out, (
            f"[{name}] rep {rep}: the bind was served from cache, so `b` is a "
            f"deserialised copy and no longer the live object: {out!r}"
        )


def _live(*lines: str) -> str:
    """A probe cell whose EVERY statement is individually ``@cash:no-cache``."""
    return "\n".join(f"# @cash:no-cache\n{line}" for line in lines)


def _restart(nb_runner):
    nb_runner.restart()
    nb_runner._inject_notebook_path()


def _cells(bind, id_expr, val_expr):
    return [
        SETUP,                                              # 1
        BUILD,                                              # 2
        f"# @cash:persist\n{bind}",                         # 3
        MUTATE,                                             # 4
        _live(f"print('same', {id_expr})",
              f"print('val', {val_expr})"),                 # 5
    ]


@pytest.mark.parametrize("name,bind,id_expr,val_expr,expected", FORMS)
def test_computed_rhs_alias_identity_holds(nb_runner, name, bind, id_expr, val_expr, expected):
    """``{id_expr}`` must hold on every re-run, as it does in plain Python."""
    nb_runner.create_notebook(_cells(bind, id_expr, val_expr))
    nb_runner.start_kernel()

    failures = []
    for rep in range(WARM_REPS):
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "RESTORED" not in out, (
            f"rep {rep}: the probe cell replayed from cache, so it is NOT reading "
            f"the live namespace -- this probe is invalid, not passing: {out!r}"
        )
        bind_out = nb_runner.get_output(3)
        for key, want in (("same", "same True"), ("val", f"val {expected}")):
            if want not in out:
                got = next(
                    (ln for ln in out.splitlines() if ln.startswith(key + " ")),
                    "<line missing>",
                )
                failures.append(
                    f"[{name}] rep {rep}: expected {want!r}, got {got!r} "
                    f"(bind cell: {'RESTORED' if 'RESTORED' in bind_out else 'executed'})"
                )

    assert not failures, "CAS-188 REPRODUCES:\n" + "\n".join(failures)


@pytest.mark.parametrize("name,bind,id_expr,val_expr,expected", FORMS)
def test_computed_rhs_alias_matches_plain_kernel(tmp_path, name, bind, id_expr, val_expr, expected):
    """Ground truth: cash ON must print exactly what a cash-OFF kernel prints."""
    cells_off = [BUILD, bind, MUTATE,
                 f"print('same', {id_expr})\nprint('val', {val_expr})"]

    def _run(work_dir, cells, with_cash, probe_cell):
        work_dir.mkdir(parents=True, exist_ok=True)
        runner = NotebookTestRunner(work_dir=work_dir, use_pool=False)
        try:
            runner.create_notebook(cells)
            runner.start_kernel(with_cash=with_cash)
            for _ in range(WARM_REPS):
                runner.run_all()
            return runner.get_output(probe_cell)
        finally:
            runner.shutdown()

    def _probe_lines(raw: str) -> str:
        """The probe's printed lines, free of stream framing.

        Applied to BOTH sides. ipykernel buffers stdout and flushes on a timer,
        so two `print` calls arrive as one stream message or two depending on
        machine speed; joining them inserts a blank line in the second case.
        Comparing raw output made the oracle guard fail on slower machines with
        a spurious blank line, which reads as a cash bug and is not one. The
        badge is cash-only chrome and is filtered by the same rule.
        """
        return "\n".join(
            ln for ln in raw.splitlines() if ln.startswith(("same ", "val "))
        ).strip()

    off = _probe_lines(_run(tmp_path / "off", cells_off, False, 4))
    on_lines = _probe_lines(_run(tmp_path / "on", _cells(bind, id_expr, val_expr), True, 5))

    assert off == f"same True\nval {expected}", (
        f"[{name}] the cash-off oracle itself is wrong -- test bug: {off!r}"
    )
    assert on_lines == off, (
        f"[{name}] cash ON disagrees with a plain kernel on `{bind}`.\n"
        f"--- cash OFF ---\n{off!r}\n--- cash ON ---\n{on_lines!r}"
    )


def test_computed_rhs_alias_identity_after_restart(nb_runner):
    """The stale copy also comes back from DISK after a real kernel restart."""
    name, bind, id_expr, val_expr, expected = FORMS[0]
    nb_runner.create_notebook(_cells(bind, id_expr, val_expr))
    nb_runner.start_kernel()
    nb_runner.run_all()

    _restart(nb_runner)

    failures = []
    for rep in range(3):
        nb_runner.run_all()
        out = nb_runner.get_output(5)
        assert "RESTORED" not in out, f"rep {rep}: probe cell replayed: {out!r}"
        if "same True" not in out or f"val {expected}" not in out:
            failures.append(f"post-restart rep {rep}: {out.splitlines()[:2]!r}")

    assert not failures, "CAS-188 REPRODUCES after a kernel restart:\n" + "\n".join(failures)

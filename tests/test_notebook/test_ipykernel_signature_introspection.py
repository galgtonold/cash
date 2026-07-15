"""ipykernel introspects the shell's execution hooks — cash must not lie to it (CAS-134).

ipykernel decides whether to pass ``cell_id`` to ``shell.run_cell_async`` /
``shell.run_cell`` by introspecting their signatures:

    ipykernel/ipkernel.py:395   accepts_params = _accepts_parameters(run_cell, ["cell_id"])
    ipykernel/kernelbase.py:76  parameters = inspect.signature(meth).parameters
                                ... or any(p.kind == p.VAR_KEYWORD for p in parameters.values())

That last clause is the trap: a wrapper declared ``(*args, **kwargs)`` answers
**True for every parameter name**, whether or not the wrapped callee accepts it.
So when cash patches these hooks it must present the *original's* signature —
otherwise ipykernel is told "accepts anything", passes ``cell_id=...``, cash
forwards it to an original that rejects it, and the resulting ``TypeError``
escapes before ``execute_reply`` is sent: the cell hangs at ``[*]`` forever.

That combination is reachable on cash's own supported floor: ``[notebook]``
allows ``ipython>=8.0``, and IPython only grew ``cell_id`` on ``run_cell_async``
in **8.3** — so IPython 8.0/8.1/8.2 + any ipykernel that introspects (6.x and
7.x both do) is a bricked kernel.

These tests run ipykernel's REAL introspection helper rather than a
reimplementation of it, and assert *parity*: the verdict on cash's patched hook
must be identical to the verdict on the untouched original. Parity is the
property that matters, and it is version-independent — it holds no matter which
parameters ipykernel decides to probe for in the future.
"""
import inspect

import pytest
from traitlets.config import Configurable

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.ipython.magics import CashMagics

# ipykernel is a dev/runtime companion, not a hard dependency of the [notebook]
# extra — skip rather than fail the bare-install matrix.
kernelbase = pytest.importorskip("ipykernel.kernelbase")
_accepts_parameters = kernelbase._accepts_parameters

# The names ipykernel probes for today, plus a name it will never probe for.
# The made-up name is the actual lie detector: an honest signature says False,
# a ``**kwargs`` signature says True. This makes the test catch the bug on every
# IPython version, including ones whose real signature already accepts cell_id.
PROBE = ["cell_id", "cell_meta", "_cash_definitely_not_a_real_parameter"]


@pytest.fixture
def real_shell():
    """A real IPython shell. Cleared from the process-global singleton on exit."""
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    try:
        yield shell
    finally:
        InteractiveShell.clear_instance()


def _patch_cash_onto(shell):
    """Instantiate CashMagics, which monkey-patches the shell's exec hooks."""
    return CashMagics(shell, Cash(backend=InMemoryBackend(), register_magic=False))


class TestIntrospectionParity:
    """cash's patched hooks must answer introspection exactly as the originals do."""

    def test_run_cell_async_verdict_matches_original(self, real_shell):
        original = real_shell.run_cell_async
        before = _accepts_parameters(original, PROBE)

        _patch_cash_onto(real_shell)

        assert real_shell.run_cell_async is not original, "precondition: cash patched the hook"
        after = _accepts_parameters(real_shell.run_cell_async, PROBE)
        assert after == before, (
            "cash's run_cell_async wrapper reports a different set of accepted "
            f"parameters than the original ({after} != {before}). ipykernel uses "
            "exactly this verdict to decide what to pass — lying makes it pass "
            "arguments the real callee rejects, and the cell hangs at [*]."
        )

    def test_run_cell_verdict_matches_original(self, real_shell):
        original = real_shell.run_cell
        before = _accepts_parameters(original, PROBE)

        _patch_cash_onto(real_shell)

        assert real_shell.run_cell is not original, "precondition: cash patched the hook"
        after = _accepts_parameters(real_shell.run_cell, PROBE)
        assert after == before, (
            "cash's run_cell wrapper reports a different set of accepted parameters "
            f"than the original ({after} != {before}). ipykernel introspects this "
            "hook too (ipkernel.py:404) on shells without run_cell_async."
        )

    @pytest.mark.parametrize("hook", ["run_cell", "run_cell_async"])
    def test_patched_hook_exposes_original_signature(self, real_shell, hook):
        """inspect.signature() — what ipykernel calls — must see the real signature."""
        before = inspect.signature(getattr(real_shell, hook))
        _patch_cash_onto(real_shell)
        after = inspect.signature(getattr(real_shell, hook))
        assert str(after) == str(before), (
            f"shell.{hook} signature changed under cash: {after} != {before}"
        )


class LegacyShell(Configurable):
    """Shell whose hooks carry IPython<8.3 signatures (no ``cell_id``).

    This is the exact configuration CAS-134 bricks: the original rejects
    ``cell_id``, so if cash's wrapper claims to accept it, ipykernel passes it
    and the forward blows up with TypeError.
    """

    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.user_global_ns = self.user_ns
        self.input_transformers_cleanup = []
        self.ast_transformers = []
        self.events = _NullEvents()
        self.display_pub = type("P", (), {"publish": lambda *a, **k: None})()
        self.calls = []

    # Signatures copied verbatim from IPython 8.2.0's interactiveshell.py.
    def run_cell(self, raw_cell, store_history=False, silent=False, shell_futures=True):
        self.calls.append(raw_cell)
        return "sync-result"

    async def run_cell_async(
        self,
        raw_cell: str,
        store_history=False,
        silent=False,
        shell_futures=True,
        *,
        transformed_cell=None,
        preprocessing_exc_tuple=None,
    ):
        self.calls.append(raw_cell)
        return "async-result"


class _NullEvents:
    def register(self, *a, **k):
        pass

    def unregister(self, *a, **k):
        pass

    def trigger(self, *a, **k):
        pass


class TestLegacyIPythonShell:
    """IPython 8.0-8.2 (allowed by [notebook]) + an introspecting ipykernel."""

    @pytest.mark.parametrize("hook", ["run_cell", "run_cell_async"])
    def test_cash_does_not_claim_cell_id_on_a_shell_that_rejects_it(self, hook):
        shell = LegacyShell()
        original = getattr(shell, hook)
        assert _accepts_parameters(original, ["cell_id"]) == {"cell_id": False}, (
            "precondition: the legacy signature really does reject cell_id"
        )

        _patch_cash_onto(shell)

        verdict = _accepts_parameters(getattr(shell, hook), ["cell_id"])
        assert verdict == {"cell_id": False}, (
            f"cash told ipykernel that shell.{hook} accepts cell_id, but the "
            "underlying IPython<8.3 method does not. ipykernel would pass "
            "cell_id=..., the forward would raise TypeError, no execute_reply "
            "would be sent, and the cell would hang at [*] forever (CAS-134)."
        )

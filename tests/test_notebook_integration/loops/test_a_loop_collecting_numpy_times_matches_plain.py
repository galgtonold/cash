"""A loop collecting numpy ``timedelta64`` or ``datetime64`` scalars runs as
plain Python does: the values are atoms to cash's bookkeeping, keyed by
their exact type (unit included) and raw value, never read through
``int()``, which a unit coarser than nanoseconds refuses (its ``.item()`` is
a ``datetime.timedelta``)."""

import numpy as np
import pytest

_UNITS = ["s", "ms", "us", "ns"]


def _setup(unit: str, step: int) -> str:
    return f"import numpy as np\nstamps = (np.arange(0, 400, {step}) ** 2).astype('timedelta64[{unit}]')"


_LOOP = """deltas = []
for delta in np.diff(stamps):
    deltas.append(delta)"""
_SHOW = "print(len(deltas), deltas[0].dtype, [int(d.astype('int64')) for d in deltas[:5]], sum(deltas[1:], deltas[0]))"


#: What the kernel holds, read past any replayed output.
_STATE = "(len(deltas), sorted({str(d.dtype) for d in deltas}), [int(d.astype('int64')) for d in deltas[-3:]])"


def _plain_state(unit: str, step: int) -> str:
    stamps = (np.arange(0, 400, step) ** 2).astype(f"timedelta64[{unit}]")
    deltas = list(np.diff(stamps))
    return repr((len(deltas), [f"timedelta64[{unit}]"], [int(d.astype("int64")) for d in deltas[-3:]]))


def _plain(unit: str, step: int) -> str:
    stamps = (np.arange(0, 400, step) ** 2).astype(f"timedelta64[{unit}]")
    deltas = list(np.diff(stamps))
    return (
        f"{len(deltas)} {deltas[0].dtype} {[int(d.astype('int64')) for d in deltas[:5]]} {sum(deltas[1:], deltas[0])}"
    )


@pytest.mark.loops
@pytest.mark.parametrize("unit", _UNITS)
def test_a_loop_collecting_timedeltas_matches_plain(nb_runner, unit):
    """Each unit: the first run, a re-run and a run after a restart all print
    what plain Python prints, with no error."""
    nb_runner.create_notebook([_setup(unit, 3), _LOOP, _SHOW])
    nb_runner.start_kernel()
    nb_runner.run_all()
    expected = _plain(unit, 3)
    for again in (False, True, "restart"):
        if again == "restart":
            nb_runner.restart()
        if again:
            nb_runner.run_all()
        assert nb_runner.get_output(3).strip() == expected
        assert nb_runner.peek(_STATE).strip() == _plain_state(unit, 3)


@pytest.mark.loops
def test_a_loop_collecting_datetimes_matches_plain(nb_runner):
    """``datetime64`` at a unit coarser than nanoseconds, the same way."""
    cells = [
        "import numpy as np\nstamps = np.datetime64('2024-01-01T00:00:00', 's') + np.arange(0, 300, 7).astype('timedelta64[s]')",
        """moments = []
for moment in stamps:
    moments.append(moment)""",
        "print(len(moments), moments[0].dtype, moments[0], moments[-1])",
    ]
    stamps = np.datetime64("2024-01-01T00:00:00", "s") + np.arange(0, 300, 7).astype("timedelta64[s]")
    expected = f"{len(stamps)} {stamps[0].dtype} {stamps[0]} {stamps[-1]}"
    state = repr((len(stamps), str(stamps[-1].dtype), str(stamps[-1])))
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    for again in (False, True, "restart"):
        if again == "restart":
            nb_runner.restart()
        nb_runner.run_all()
        assert nb_runner.get_output(3).strip() == expected
        assert nb_runner.peek("(len(moments), str(moments[-1].dtype), str(moments[-1]))").strip() == state


@pytest.mark.loops
def test_a_changed_timedelta_value_or_unit_is_followed(nb_runner):
    """The values key by type and value: an edit to the data, or to just
    its unit (the same integers), reruns the loop and shows the new result."""
    nb_runner.create_notebook([_setup("s", 3), _LOOP, _SHOW])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == _plain("s", 3)
    nb_runner.set_cell_source(1, _setup("s", 5))
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == _plain("s", 5)
    assert nb_runner.peek(_STATE).strip() == _plain_state("s", 5)
    nb_runner.set_cell_source(1, _setup("ms", 5))
    nb_runner.run_all()
    assert nb_runner.get_output(3).strip() == _plain("ms", 5)
    assert nb_runner.peek(_STATE).strip() == _plain_state("ms", 5)

"""The derivation detectors must not import numpy/pandas to ask a type question.

`detect_derivation_edges` runs on every statement. Its numpy and pandas
detectors used to `import` those libraries just to evaluate
`isinstance(value, np.ndarray)` and friends, so the FIRST cell executed under
``%cash_on`` paid the whole import: measured at 707ms for a cell that only
summed integers, ~535ms of it inside `_detect_pandas_refholder_edge`.

An ndarray or a groupby cannot exist unless its library is imported, so
`sys.modules` answers the question for free. The subtlety these tests exist
for is the NEGATIVE case: it must not be cached, because a later cell may
import pandas and the detectors have to start seeing its types from then on.
"""

import subprocess
import sys
import textwrap

import pytest


def _run(body: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout.strip()


def test_detectors_do_not_import_numpy_or_pandas():
    """A subprocess is required: pytest has already imported both."""
    out = _run(
        """
        import sys
        from cash.notebook.statement import derivation_edges as de

        assert "pandas" not in sys.modules and "numpy" not in sys.modules

        edges = {}
        for value in (42, "text", [1, 2], {"a": 1}, None):
            de.detect_derivation_edges(edges, "out", value, {})
        print("numpy" in sys.modules, "pandas" in sys.modules)
        """
    )
    assert out == "False False", f"a detector imported a heavy library: {out}"


def test_pandas_types_resolve_after_a_later_import():
    """The negative must NOT be cached -- a later cell may import pandas.

    Without this, the first statement in a session would permanently disable
    ref-holder detection for the rest of it, and a groupby holding a live
    frame would stop bumping its source's lineage.
    """
    pytest.importorskip("pandas")
    out = _run(
        """
        from cash.notebook.statement import derivation_edges as de

        before = de._pandas_refholder_types()
        import pandas  # noqa: F401
        after = de._pandas_refholder_types()
        print(len(before), len(after) > 0)
        """
    )
    assert out == "0 True", (
        "ref-holder types did not resolve after pandas was imported later; "
        "the negative result was cached"
    )


def test_numpy_view_edge_still_detected():
    """The control: gating on sys.modules must not cost the feature."""
    np = pytest.importorskip("numpy")
    from cash.notebook.statement import derivation_edges as de

    base = np.arange(12)
    view = base[2:6]
    user_ns = {"base": base, "view": view}
    edges: dict[str, set[str]] = {}
    de.detect_derivation_edges(edges, "view", view, user_ns)
    assert edges, "a numpy view produced no derivation edge"


def test_pandas_refholder_edge_still_detected():
    pd = pytest.importorskip("pandas")
    from cash.notebook.statement import derivation_edges as de

    frame = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    grouped = frame.groupby("g")
    user_ns = {"frame": frame, "grouped": grouped}
    edges: dict[str, set[str]] = {}
    de.detect_derivation_edges(edges, "grouped", grouped, user_ns)
    assert edges.get("frame") == {"grouped"}, (
        f"groupby ref-holder edge not detected: {edges}"
    )

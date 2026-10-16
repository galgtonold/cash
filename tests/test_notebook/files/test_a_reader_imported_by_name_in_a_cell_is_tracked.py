"""A watched reader a cell names directly is tracked like ``pl.read_parquet``.

The wrappers that record a read go onto the module attributes cash registered
(``polars.read_parquet``). A cell that bound the reader by name missed them in
two ways, and the statement was cached with no file recorded, so an edited
file gave the old frame without a warning:

- the name was bound before cash's first tracked statement (an import above
  ``%cash_on``, an IPython startup file), so it holds the original function;
- the name came from a submodule the library re-exports it from
  (``from polars.io.parquet import read_parquet``), which cash never wrapped.

On Linux and Windows the native file hook sees polars open the file anyway;
these tests turn it off, as on macOS, so only the wrappers can see the read.
"""

from __future__ import annotations

import pytest

from cash.tracking import native_reads
from tests._cell_driver import run_cash_cell

pl = pytest.importorskip("polars")

pytestmark = pytest.mark.core


@pytest.fixture(autouse=True)
def no_hook():
    was = native_reads.active()
    native_reads.disable()
    yield
    if was:
        native_reads.enable()


@pytest.fixture
def parquet(tmp_path):
    path = tmp_path / "d.parquet"
    pl.DataFrame({"x": [1, 2, 3]}).write_parquet(path)
    return str(path).replace("\\", "/")


def _deps(magics, name):
    return magics._statement_processor.tracking_state.executed_file_deps.get(name, set())


def test_a_reader_bound_before_cash_is_tracked(cash_magics, mock_shell, parquet):
    import polars as polars_module

    mock_shell.user_ns["read_parquet"] = polars_module.read_parquet  # bound while no tracker was open
    run_cash_cell(cash_magics, "import polars as pl")
    run_cash_cell(cash_magics, f"t = read_parquet('{parquet}')")
    assert parquet in {str(p).replace("\\", "/") for p in _deps(cash_magics, "t")}


@pytest.mark.parametrize(
    "spelling",
    [
        "from polars.io.parquet import read_parquet",
        "from polars.io.parquet.functions import read_parquet",
        "from polars import read_parquet",
    ],
)
def test_a_reader_imported_from_where_the_library_defines_it_is_tracked(cash_magics, parquet, spelling):
    run_cash_cell(cash_magics, "import polars as pl")
    run_cash_cell(cash_magics, spelling)
    run_cash_cell(cash_magics, f"t = read_parquet('{parquet}')")
    assert parquet in {str(p).replace("\\", "/") for p in _deps(cash_magics, "t")}

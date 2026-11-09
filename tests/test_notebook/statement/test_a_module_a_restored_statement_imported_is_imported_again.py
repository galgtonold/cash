"""A module a statement imported on the way, whose import registered
something with the process, is imported again when the statement is
restored.

``zzlazy.sub`` reached through the package's ``__getattr__`` (PEP 562), or
by ``from . import sub`` in a function of the package: importing it sets a
logger's level. After a restart the statement was served and the module
never imported, so the level stayed unset (`process_registries
.registering_imports`). Nor does keying the statement import it: reading
``from . import sub`` ran the package's ``__getattr__`` before the
statement, outside what it was watched for.
"""

from __future__ import annotations

import logging
import sys
import textwrap

import pytest

from tests._cell_driver import run_cash_cell

PACKAGE = "zz_lazy_pkg"


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


@pytest.fixture
def lazy_package(tmp_path):
    root = tmp_path / "mods"
    package = root / PACKAGE
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        textwrap.dedent(
            """
            import importlib

            def __getattr__(name):
                if name == "sub":
                    return importlib.import_module(".sub", __name__)
                raise AttributeError(name)

            def compute():
                from . import sub
                return sub.VALUE
            """
        ),
        encoding="utf-8",
    )
    (package / "sub.py").write_text(
        "import logging\nlogging.getLogger('zz.lazy').setLevel(10)\nVALUE = 7\n", encoding="utf-8"
    )
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path.remove(str(root))
        for name in (PACKAGE, f"{PACKAGE}.sub"):
            sys.modules.pop(name, None)
        logging.getLogger("zz.lazy").setLevel(0)


def _fresh():
    """The submodule and its setting as a new process finds them."""
    sys.modules.pop(f"{PACKAGE}.sub", None)
    package = sys.modules.get(PACKAGE)
    if package is not None:
        vars(package).pop("sub", None)
    logging.getLogger("zz.lazy").setLevel(0)


@pytest.mark.usefixtures("store_everything", "lazy_package")
@pytest.mark.parametrize(
    "statement",
    [f"v = {PACKAGE}.sub.VALUE * 2", f"v = {PACKAGE}.compute() * 2"],
    ids=["package-getattr", "from-dot-import-in-a-function"],
)
def test_the_module_is_imported_again_after_a_restart(statement, cash_magics):
    cells = [f"import {PACKAGE}", statement]
    _fresh()
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert logging.getLogger("zz.lazy").level == 10
    cash_magics.tracking_state.reset_session_state()
    cash_magics._upstream_checker.simulator.cache.reset()
    _fresh()
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert f"{PACKAGE}.sub" in sys.modules, statuses
    assert logging.getLogger("zz.lazy").level == 10, statuses


def _reading(read) -> bool:
    """Whether *read*, cash reading code that names the submodule, imported it."""
    import importlib

    importlib.import_module(PACKAGE)
    _fresh()
    read()
    return f"{PACKAGE}.sub" in sys.modules


@pytest.mark.usefixtures("lazy_package")
def test_reading_a_body_import_does_not_run_the_package_s_getattr():
    from cash.analysis.helper_bindings import resolve_local_import
    from cash.analysis.unloaded_modules import unloaded_import

    assert not _reading(lambda: resolve_local_import(PACKAGE, ("sub",)))
    assert not _reading(lambda: unloaded_import(PACKAGE, ("sub",), None))


@pytest.mark.usefixtures("lazy_package")
def test_reading_a_statement_s_import_does_not_run_the_package_s_getattr():
    import ast

    from cash.analysis import code_analyzer

    def read():
        visitor = code_analyzer._ForbiddenVisitor({})
        visitor.visit(ast.parse(f"from {PACKAGE} import sub"))

    assert not _reading(read)


@pytest.mark.usefixtures("store_everything", "lazy_package")
def test_the_module_is_imported_again_after_a_logger_was_looked_up(cash_magics):
    """A statement above found a logger by its name, so the loggers calls
    return are watched. The one the module's import set is the module's,
    as the process registries it changes are: the statement is stored on
    its first run, which imported the module, and the hit imports it."""
    cells = [
        "import logging\nlv = logging.getLogger('zz.looked_up').level",
        f"import {PACKAGE}",
        f"v = {PACKAGE}.compute() * 2",
    ]
    _fresh()
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert statuses == ["COMPUTED"], statuses
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert statuses == ["RESTORED"], statuses
    cash_magics.tracking_state.reset_session_state()
    cash_magics._upstream_checker.simulator.cache.reset()
    _fresh()
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert f"{PACKAGE}.sub" in sys.modules
    assert logging.getLogger("zz.lazy").level == 10

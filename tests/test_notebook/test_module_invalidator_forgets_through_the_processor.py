"""A module edit drops a dependent's lineage through one processor method.

``ModuleInvalidator`` used to pop the processor's lineage maps itself, a
different subset at each of four sites, and to print its diagnostics to
stdout only if ``debug`` was set when it was built. It now forgets a variable
through :meth:`StatementProcessor.forget_variable` and logs at debug level.
"""

import logging
import types
from unittest.mock import MagicMock

import pytest

from cash.backends import FileBackend
from cash.core import Cash
from cash.notebook.module_invalidator import ModuleInvalidator
from cash.notebook.statement import StatementProcessor
from cash.notebook.tracking_state import TrackingState


@pytest.fixture
def processor(tmp_path):
    shell = MagicMock()
    shell.user_ns = {}
    return StatementProcessor(
        cash_instance=Cash(backend=FileBackend(str(tmp_path)), register_magic=False),
        shell=shell,
        tracking_state=TrackingState(),
    )


def _record(proc, name):
    proc.tracking_state.lineage.record(name, "lin")
    proc.tracking_state.executed_cell_codes[name] = f"{name} = lib.f()"
    proc.tracking_state.executed_input_lineages[name] = {"lib": "old"}
    proc.tracking_state.current_session_hashes[name] = "h"
    proc.tracking_state.from_import_components[name] = "c"
    proc.tracking_state.module_attribute_deps[name] = {"lib": {"f"}}


def _recorded(proc, name):
    state = proc.tracking_state
    return {
        "variable_lineage": name in state.variable_lineage,
        "executed_cell_codes": name in state.executed_cell_codes,
        "executed_input_lineages": name in state.executed_input_lineages,
        "current_session_hashes": name in state.current_session_hashes,
        "from_import_components": name in state.from_import_components,
        "module_attribute_deps": name in state.module_attribute_deps,
    }


def test_forget_variable_drops_every_record_of_the_name(processor):
    _record(processor, "y")
    _record(processor, "keep")

    processor.forget_variable("y")

    assert not any(_recorded(processor, "y").values()), _recorded(processor, "y")
    assert all(_recorded(processor, "keep").values()), "another name's records went too"


def test_a_dependent_is_invalidated_and_the_step_is_logged_not_printed(processor, caplog, capsys):
    _record(processor, "y")
    invalidator = ModuleInvalidator(types.SimpleNamespace(user_ns={"y": 1}))

    with caplog.at_level(logging.DEBUG, logger="cash.notebook.module_invalidator"):
        invalidator._propagate_module_invalidation({"lib": "old"}, {}, processor)

    assert not any(_recorded(processor, "y").values()), _recorded(processor, "y")
    assert "y" in processor.tracking_state.rerun_bindings
    assert "Cleared lineage for dependent var 'y'" in caplog.text
    assert capsys.readouterr().out == ""

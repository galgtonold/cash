"""`sc.pp.calculate_qc_metrics(adata, inplace=True)` does not write `sc`.

Round 28, r28s4: the badge said these lines "Produced sc" -- the module.
``inplace=True`` makes the analysis treat the call's receiver as mutated, and
for a function reached through a module the receiver chain is rooted at the
module name. At runtime that bumped ``sc``'s lineage on every such line, so
every statement reading ``sc`` missed; and it said nothing about ``adata``,
the object actually changed (that one is observed at runtime since 585d9ec).

A statement that does not import or assign a name cannot produce a module.
Module SETTINGS (``plt.rcParams.update``, ``pd.set_option``) are a separate,
deliberate route and are unaffected.
"""

import ast
import types

from cash.notebook.analysis import CodeAnalyzer


def _ns():
    sc = types.ModuleType("scanpy")
    sc.pp = types.SimpleNamespace(calculate_qc_metrics=lambda *a, **k: None)
    return {"sc": sc, "adata": object(), "np": types.ModuleType("numpy")}


def test_an_inplace_call_through_a_module_does_not_output_the_module():
    code = "sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)"
    _inputs, outputs = CodeAnalyzer.analyze_code_block(code, tree=ast.parse(code), user_ns=_ns())
    assert "sc" not in outputs, outputs


def test_an_import_still_outputs_the_module():
    code = "import numpy as np"
    _inputs, outputs = CodeAnalyzer.analyze_code_block(code, tree=ast.parse(code), user_ns=_ns())
    assert "np" in outputs, outputs


def test_a_method_inplace_on_a_frame_still_outputs_the_frame():
    code = "df.sort_values('a', inplace=True)"
    ns = {**_ns(), "df": object()}
    _inputs, outputs = CodeAnalyzer.analyze_code_block(code, tree=ast.parse(code), user_ns=ns)
    assert "df" in outputs, outputs

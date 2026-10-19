"""A code object's source, read from its file on disk without
``inspect.getmodule`` (`source_reading._getsourcelines`), is what
``inspect.getsourcelines`` answers.

A code object names no module, so ``getmodule`` resolved the file of every
module loaded since it last looked: 105 ms of the first cell using ``open``
(IPython's ``_modified_open``). What must hold: for every function and code
object reachable from a sample of loaded modules, the same lines and first
line as ``inspect`` gives, or ``inspect`` asked; and a cell using ``open``
keyed as before.
"""

from __future__ import annotations

import inspect
import types

from cash import source_reading
from tests._cell_driver import run_cash_cell

SAMPLE = [
    "json",
    "csv",
    "os",
    "inspect",
    "collections",
    "re",
    "functools",
    "pathlib",
    "glob",
    "IPython.core.interactiveshell",
]


def _codes_of(module) -> list[types.CodeType]:
    """Every code object in *module*'s functions and classes, nested ones too."""
    seen: dict[int, types.CodeType] = {}
    stack = []
    for value in vars(module).values():
        if inspect.isclass(value) and getattr(value, "__module__", None) == module.__name__:
            stack.extend(getattr(v, "__code__", None) for v in vars(value).values())
            stack.extend(getattr(getattr(v, "__func__", None), "__code__", None) for v in vars(value).values())
        elif inspect.isfunction(value) and value.__module__ == module.__name__:
            stack.append(value.__code__)
    while stack:
        code = stack.pop()
        if not isinstance(code, types.CodeType) or id(code) in seen:
            continue
        seen[id(code)] = code
        stack.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
    return list(seen.values())


def _modules() -> list:
    """The stdlib sample, and every loaded module of numpy, pandas and IPython."""
    import importlib
    import sys

    for name in ("numpy", "pandas", "IPython"):
        try:
            importlib.import_module(name)
        except ImportError:
            pass
    out = [importlib.import_module(name) for name in SAMPLE]
    out += [
        m
        for n, m in sorted(sys.modules.items())
        if n.partition(".")[0] in ("numpy", "pandas", "IPython") and m is not None
    ]
    return out


def _inspect_answer(code):
    try:
        return inspect.getsourcelines(code)
    except Exception as exc:  # whatever inspect raises, raised alike
        return type(exc)


#: Code objects compared, every n-th of the sample's: all of them took ~7 s
#: alone, past the per-test timeout under a loaded run.
STRIDE = 6


def test_every_code_object_in_the_sample_reads_as_inspect_reads_it():
    compared = read_on_disk = 0
    codes = [(module, code) for module in _modules() for code in _codes_of(module)]
    for module, code in codes[::STRIDE]:
        expected = _inspect_answer(code)
        fast = source_reading._code_source_lines_on_disk(code)
        if fast is not None:
            read_on_disk += 1
            assert fast == expected, (module.__name__, code)
        try:
            got = source_reading._getsourcelines(code)
        except Exception as exc:  # whatever inspect raises, raised alike
            got = type(exc)
        assert got == expected, (module.__name__, code)
        compared += 1
    assert read_on_disk > 1500 and read_on_disk > compared // 2, (read_on_disk, compared)


def test_ipython_s_open_reads_as_inspect_reads_it():
    from IPython.core.interactiveshell import _modified_open

    code = _modified_open.__code__
    assert source_reading._code_source_lines_on_disk(code) is not None, "control: read from disk"
    assert source_reading._getsourcelines(code) == inspect.getsourcelines(code)


def _cell_key_parts(mock_shell, cash_instance, cell: str, monkeypatch) -> tuple:
    """What a fresh ``%cash_on`` session keys *cell* on that a function's
    source decides -- its callables' component, and ``text``'s lineage --
    with no earlier source read, or entry, remembered."""
    from cash.notebook.ipython.magics import CashMagics
    from cash.notebook.statement import lineage

    cash_instance.backend.clear()
    source_reading._SOURCE_LINES.clear()
    source_reading._HELD_LINES.clear()
    components: list[str] = []
    real = lineage.callable_source_component
    with monkeypatch.context() as patches:
        patches.setattr(lineage, "callable_source_component", lambda *a: components.append(real(*a)) or components[-1])
        magics = CashMagics(mock_shell, cash_instance)
        run_cash_cell(magics, cell)
    assert mock_shell.user_ns["text"] == "some text"
    assert any(":open:" in c for c in components), components
    return components, magics.tracking_state.variable_lineage["text"]


def test_a_cell_using_open_keys_as_it_did_through_inspect(mock_shell, cash_instance, tmp_path, monkeypatch):
    from IPython.core.interactiveshell import _modified_open

    monkeypatch.chdir(tmp_path)
    (tmp_path / "f.txt").write_text("some text", encoding="utf-8")
    mock_shell.user_ns["open"] = _modified_open
    cell = "text = open('f.txt').read()"
    with monkeypatch.context() as patches:
        patches.setattr(source_reading, "_code_source_lines_on_disk", lambda code: None)
        through_inspect = _cell_key_parts(mock_shell, cash_instance, cell, monkeypatch)
    read_on_disk = []
    real = source_reading._code_source_lines_on_disk
    monkeypatch.setattr(
        source_reading, "_code_source_lines_on_disk", lambda code: read_on_disk.append(code) or real(code)
    )
    assert _cell_key_parts(mock_shell, cash_instance, cell, monkeypatch) == through_inspect
    assert _modified_open.__code__ in read_on_disk, "control: the cell's open was read from disk"

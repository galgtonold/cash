"""Every call that runs a magic, a shell command or code through the shell is
read as one, and so runs every time (`calls_ipython`).

IPython's transform writes ``get_ipython().run_line_magic(...)``, but
notebooks saved before IPython 9 call the deprecated alias ``magic``, and code
reaches a runner with ``getattr`` or through ``IPython.get_ipython()``. Those
were read as plain calls and cached: ``get_ipython().magic('ls ')`` showed the
first folder's listing after a ``%cd``.
"""

from __future__ import annotations

import pytest

from cash.analysis.code_analyzer import CodeAnalyzer, calls_ipython


@pytest.mark.parametrize(
    "code",
    [
        "get_ipython().run_line_magic('ls', '')",
        "get_ipython().magic('ls ')",
        "files = get_ipython().magic('sx ls')",
        "getattr(get_ipython(), 'run_line_magic')('ls', '')",
        "getattr(get_ipython(), 'magic')('ls')",
        "getattr(get_ipython(), 'system')('ls')",
        "import IPython\nIPython.get_ipython().run_line_magic('ls', '')",
        "get_ipython().system_piped('ls')",
        "get_ipython().run_cell('%ls')",
        "get_ipython().ex('import os')",
        "for d in dirs:\n    get_ipython().magic(f'cd {d}')",
    ],
)
def test_it_runs_through_the_shell(code):
    assert calls_ipython(CodeAnalyzer.parse_cell(code))


@pytest.mark.parametrize(
    "code",
    [
        "ns = get_ipython().user_ns",
        "getattr(get_ipython(), 'user_ns')",
        "parser.magic('ls')",
        "getattr(parser, 'magic')('ls')",
        "get_ipython",
    ],
)
def test_other_calls_do_not(code):
    assert not calls_ipython(CodeAnalyzer.parse_cell(code))

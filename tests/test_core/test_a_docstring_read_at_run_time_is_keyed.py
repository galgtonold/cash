"""A docstring the code reads at run time is an input; otherwise it is not.

Docstrings stay out of the key (``test_docstring_identity.py``), so editing
one keeps the stored results. But code that READS one -- a tool description,
a prompt or help text built from ``f.__doc__``, ``tool.__doc__``,
``C.__doc__`` or ``inspect.getdoc(g)`` -- computes with it, and editing it
served the old answer. Such code keys the docstrings it can reach.

Fresh process per run.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = pytest.mark.core

TOOLS = textwrap.dedent('''
    import inspect

    def search(q):
        """{doc} search"""
        return q

    class Calc:
        """{doc} calc"""

    def describe():
        return inspect.getdoc(search) + "|" + Calc.__doc__
''')

MAIN = textwrap.dedent('''
    import json, sys, time, warnings
    warnings.simplefilter("ignore")
    import cash, tools

    def body():
        print("RAN", file=sys.stderr)  # @cash:assume-safe
        time.sleep(0.2)

    @cash.cache
    def own(x):
        """{doc} own"""
        body()
        return own.__doc__ + str(x)

    @cash.cache
    def via_helper(x):
        body()
        return tools.describe() + str(x)

    @cash.cache
    def plain(x):
        """{doc} not read"""
        body()
        return x

    print(json.dumps([own(1), via_helper(1), plain(1)]))
''')


def _run(tmp_path, doc, plain_doc=None):
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    (proj / "tools.py").write_text(TOOLS.format(doc=doc), encoding="utf-8")
    main = MAIN.format(doc=doc).replace(f"{doc} not read", f"{plain_doc or 'same'} not read")
    (proj / "main.py").write_text(main, encoding="utf-8")
    out = run_python("main.py", cwd=proj, cache_dir=tmp_path / "cache")
    return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr.count("RAN")


def test_editing_a_docstring_the_code_reads_recomputes(tmp_path):
    """THE BUG: `alpha` -> `beta` served `alpha1`."""
    assert _run(tmp_path, "alpha") == (["alpha own1", "alpha search|alpha calc1", 1], 3)
    assert _run(tmp_path, "beta") == (["beta own1", "beta search|beta calc1", 1], 2)


def test_a_docstring_nothing_reads_stays_out_of_the_key(tmp_path):
    """The control: editing a docstring no code reads keeps the entry."""
    _run(tmp_path, "alpha", plain_doc="first")
    assert _run(tmp_path, "alpha", plain_doc="second") == (["alpha own1", "alpha search|alpha calc1", 1], 0)

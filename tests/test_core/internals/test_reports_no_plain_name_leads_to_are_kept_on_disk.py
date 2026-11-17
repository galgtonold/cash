"""Purity reports a plain name does not lead back to are still stored, and a
fresh process reads them instead of walking again; an edit still walks.

A report is stored only when every function the walk visited can be found
again in the next process. Three kinds were not, so their walks ran in
every process, hits included (a few hundred ms on scanpy's first hit):

* a ``functools.singledispatch`` implementation registered as ``def _``,
  which the module's ``_`` no longer holds once another is registered;
  found again through the dispatcher's registry, by the class it is
  registered for, or by its place there when no name leads to that class
  (scanpy registers one for a stand-in of dask's array when dask is not
  installed);
* a function a decorator turned into another object (``typing.Literal``
  is a ``_SpecialForm`` around its function), analysed when a call reaches
  it as data: its report is found by where its code is written;
* a function a factory set on a class under the name ``functools.wraps``
  copied from another class's method (scanpy's ``_LegacyRng``): found
  through the class that holds it.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from tests._scripts import run_python

_RUNNER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    import cash, app, lib
    from cash.analysis import report_store
    from cash.analysis.purity_analyzer import get_analyzer
    an = get_analyzer()
    out = {}
    for name, fn in (("top", app.top), ("pick", lib.Pick._getitem)):
        before = report_store.status()
        an.analyze(fn)
        out[name] = {k: v - before[k] for k, v in report_store.status().items()}
    top = cash.cache(app.top)
    value = top([1, 2])
    key = top.explain([1, 2]).cache_key
    report_store.flush()
    print(json.dumps({"stats": report_store.status(), "of": out, "key": key, "value": value}))
    """
)

_APP = """
import lib


def top(xs):
    return [lib.double(xs), lib.double(tuple(xs)), lib.Pick[1]]
"""

_LIB = """
from functools import singledispatch


def _scale():
    return 3


#: A class no name leads to: a stand-in for a library that is not installed.
Missing = type("Array", (), {})
Missing.__module__ = "not_installed.array"
_STAND_IN = Missing
del Missing


@singledispatch
def double(x):
    return x * 2


@double.register(list)
def _(x):
    return [v * _scale() for v in x]


@double.register(_STAND_IN)
def _(x):
    return _scale()


@double.register(tuple)
def _(x):
    return tuple(v * 2 for v in x)


class _Form:
    def __init__(self, getitem):
        self._getitem = getitem

    def __getitem__(self, parameters):
        return self._getitem(self, parameters)


def _form(getitem):
    return _Form(getitem)


@_form
def Pick(self, parameters):
    return _scale() + parameters
"""


def _run(code_dir: Path, verify: bool = False) -> dict:
    env = {"CASH_ANALYSIS_MEMO_VERIFY": "1" if verify else None}
    out = run_python("-c", _RUNNER, str(code_dir), cwd=code_dir, cache_dir=code_dir / ".cash", env=env)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _write(code: Path, lib: str = _LIB) -> None:
    code.mkdir(exist_ok=True)
    (code / "lib.py").write_text(lib, encoding="utf-8")
    (code / "app.py").write_text(_APP, encoding="utf-8")


def test_a_dispatch_impl_and_a_decorated_function_are_read_back_in_a_new_process(tmp_path):
    code = tmp_path / "code"
    _write(code)
    first = _run(code)
    assert first["value"] == [[3, 6], [2, 4], 4]
    assert not first["of"]["top"]["loaded"] and not first["of"]["pick"]["loaded"], first["of"]
    second = _run(code)
    # Neither root is walked again: both reports come back from disk.
    for name in ("top", "pick"):
        assert second["of"][name]["loaded"] == 1 and second["of"][name]["missed"] == 0, (name, second["of"])
    assert (second["key"], second["value"]) == (first["key"], first["value"])
    assert _run(code, verify=True)["stats"]["verified"] >= 2


def test_an_edit_to_a_dispatch_impl_walks_again_and_misses(tmp_path):
    code = tmp_path / "code"
    _write(code)
    first = _run(code)
    assert _run(code)["of"]["top"]["loaded"] == 1
    _write(code, _LIB.replace("return [v * _scale() for v in x]", "return [v * _scale() + 1 for v in x]"))
    edited = _run(code)
    assert edited["of"]["top"]["stale"] == 1 and not edited["of"]["top"]["loaded"], edited["of"]
    assert edited["value"] == [[4, 7], [2, 4], 4]
    assert edited["key"] != first["key"]


def test_an_edit_to_the_impl_registered_for_a_stand_in_class_walks_again(tmp_path):
    code = tmp_path / "code"
    _write(code)
    first = _run(code)
    _run(code)
    _write(code, _LIB.replace("def _(x):\n    return _scale()\n", "def _(x):\n    return _scale() * 2\n"))
    edited = _run(code)
    assert edited["of"]["top"]["stale"] == 1 and not edited["of"]["top"]["loaded"], edited["of"]
    assert edited["key"] != first["key"]


def test_an_edit_to_a_decorated_function_walks_it_again(tmp_path):
    code = tmp_path / "code"
    _write(code)
    first = _run(code)
    assert _run(code)["of"]["pick"]["loaded"] == 1
    _write(code, _LIB.replace("return _scale() + parameters", "return _scale() - parameters"))
    edited = _run(code)
    # Its own text changed: no stored entry is for it any more.
    assert edited["of"]["pick"]["missed"] == 1 and not edited["of"]["pick"]["loaded"], edited["of"]
    assert edited["value"] == [[3, 6], [2, 4], 2]
    assert edited["key"] != first["key"]


def test_a_closure_is_not_found_by_where_its_code_is_written(tmp_path):
    """Two closures of one factory share their code and text: only the
    function object tells them apart, so neither is stored by its place."""
    from cash.analysis import report_store

    def make(step):
        def run(x):
            return step(x)

        return run

    assert report_store._code_root(make(abs)) is None
    assert report_store._code_root(_module_level) is not None


def _module_level(x):
    return x


def test_a_method_a_factory_named_after_another_class_is_found_through_its_class(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "legacy_rng.py").write_text(
        textwrap.dedent(
            """
            from functools import wraps


            class Source:
                def shuffle(self, x):
                    return x


            class Legacy:
                pass


            def _delegate(cls):
                for name in ("shuffle",):
                    meth = getattr(Source, name)

                    def mk(meth):
                        @wraps(meth)
                        def wrapper(self, *args):
                            return meth(self, *args)

                        return wrapper

                    setattr(cls, name, mk(meth))


            _delegate(Legacy)
            """
        ),
        encoding="utf-8",
    )
    sys.path.insert(0, str(code))
    try:
        import legacy_rng

        from cash.analysis import report_store

        wrapper = vars(legacy_rng.Legacy)["shuffle"]
        assert wrapper.__qualname__ == "Source.shuffle"
        path = report_store._path_of(wrapper)
        assert path[:2] == ["legacy_rng", ["Legacy", "shuffle"]]
        assert report_store._find(path) is wrapper
    finally:
        sys.path.remove(str(code))
        sys.modules.pop("legacy_rng", None)

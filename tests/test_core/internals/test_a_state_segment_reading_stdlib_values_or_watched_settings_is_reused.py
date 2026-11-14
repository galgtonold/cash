"""A state segment whose world holds standard-library values that cannot
change, or settings watched by reading them again, is reused.

The world watch (`cash.decorator.hit_memo._Scan`) took only atoms, code and
plain dicts; a module global holding a path, a typing form or alias, a
dataclass field or an ``lru_cache`` function kept the function's state
segment (`cash.decorator.state_memo`) from ever being remembered, and with
nilearn every call folded it again. Those are fixed once made, and what
they were built from is looked into as any value. ``abc``'s registration
record and sklearn's settings can change: they are read again on every
reuse, and a registration or ``sklearn.set_config`` builds the segment
again.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import pathlib
import sys
import textwrap
import typing

import pytest

from cash import Cash
from cash.decorator import hit_memo, state_memo

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not state_memo.ENABLED, reason="the state memo needs the C extension"),
    pytest.mark.filterwarnings("ignore::cash.exceptions.CashImpurityWarning"),
]

SOURCE = textwrap.dedent("""
    import abc
    import dataclasses
    import functools
    import pathlib
    import typing

    @dataclasses.dataclass
    class Cfg:
        size: int = 3

    class Base(abc.ABC):
        pass

    PATH = pathlib.PurePosixPath("/data/run")
    OPTIONAL = typing.Optional[int]
    PIPE = int | None
    LITERAL = typing.Literal["a", "b"]
    GENERIC = list[int]
    TYPEVAR = typing.TypeVar("T")
    ANY = typing.Any
    FIELD = dataclasses.fields(Cfg)[0]
    REGISTRY = Base._abc_impl

    @functools.lru_cache(maxsize=None)
    def twice(n):
        return 2 * n

    def uses_path(x):
        return (x, PATH.name)

    def uses_optional(x):
        return (x, repr(OPTIONAL))

    def uses_pipe(x):
        return (x, repr(PIPE))

    def uses_literal(x):
        return (x, LITERAL.__args__)

    def uses_generic(x):
        return (x, repr(GENERIC))

    def uses_typevar(x):
        return (x, TYPEVAR.__name__)

    def uses_any(x):
        return (x, repr(ANY))

    def uses_field(x):
        return (x, FIELD.default)

    def uses_lru(x):
        return twice(x)

    def uses_registry(x):
        return (x, REGISTRY is not None, isinstance(x, Base))
""")

NAMES = (
    "uses_path",
    "uses_optional",
    "uses_pipe",
    "uses_literal",
    "uses_generic",
    "uses_typevar",
    "uses_any",
    "uses_field",
    "uses_lru",
    "uses_registry",
)


@pytest.fixture
def mod(tmp_path, monkeypatch):
    name = f"stdlibvals_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield importlib.import_module(name)
    sys.modules.pop(name, None)


@pytest.fixture
def c(tmp_path):
    return Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)


def _spec(c, f):
    return c._registry.cached[f.__wrapped__.__module__ + "." + f.__wrapped__.__qualname__]


def _state(c, f):
    """The segment through the memo, and a fresh fold of it."""
    spec = _spec(c, f)
    return c._keys._code_state(spec, [], note=True), c._keys._code_state_now(spec, [], note=True)


def _reused(c, f) -> str:
    for _ in range(6):
        before = state_memo.STATS["reused"]
        got, fresh = _state(c, f)
        assert got == fresh
        if state_memo.STATS["reused"] > before:
            return got
    raise AssertionError("the segment is not being reused")


@pytest.mark.parametrize("name", NAMES)
def test_a_segment_over_a_fixed_stdlib_value_is_reused(c, mod, name):
    f = c.cache(getattr(mod, name))
    f(1)
    _reused(c, f)


@pytest.mark.parametrize(
    "value",
    [
        pathlib.PurePosixPath("/a"),
        pathlib.Path("/a"),
        pathlib.PureWindowsPath("c:/a"),
        typing.Optional,
        typing.Optional[int],
        int | str,
        typing.Literal["x"],
        typing.List[int],
        dict[str, int],
        typing.Callable[[int], str],
        typing.Annotated[int, "m"],
        typing.TypeVar("U"),
        typing.Any,
        functools.lru_cache(maxsize=None)(len),
    ],
    ids=repr,
)
def test_the_scan_takes_each_fixed_value(value):
    scan = hit_memo._Scan("m")
    scan.value(value, frozenset())
    assert not scan.probes


def test_the_scan_takes_a_dataclass_field_and_looks_into_its_default():
    @dataclasses.dataclass
    class Plain:
        size: int = 3

    @dataclasses.dataclass
    class Holding:
        rows: tuple = ([],)  # a tuple holding a list: no watcher sees the list

    hit_memo._Scan("m").value(dataclasses.fields(Plain)[0], frozenset())
    with pytest.raises(hit_memo.Ineligible):
        hit_memo._Scan("m").value(dataclasses.fields(Holding)[0], frozenset())


def test_a_path_subclass_and_a_forward_ref_are_not_taken():
    class MyPath(pathlib.PurePosixPath):
        pass

    for value in (MyPath("/a"), typing.ForwardRef("Later")):
        with pytest.raises(hit_memo.Ineligible):
            hit_memo._Scan("m").value(value, frozenset())


def test_an_abc_registration_builds_the_segment_again(c, mod):
    f = c.cache(mod.uses_registry)
    assert f(1) == (1, True, False)
    old = _reused(c, f)

    class Other:
        pass

    reused = state_memo.STATS["reused"]
    mod.Base.register(Other)
    got, fresh = _state(c, f)
    assert got == fresh
    assert state_memo.STATS["reused"] == reused  # read again, not served
    assert got != old  # the key reads the registration: a stale segment would be wrong
    assert _reused(c, f) == got


def test_sklearn_settings_are_watched(c, tmp_path, monkeypatch):
    sklearn = pytest.importorskip("sklearn")
    name = f"sklconf_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(
        textwrap.dedent("""
            from sklearn import _config

            LOCAL = _config._threadlocal

            def finite(x):
                return (x, getattr(LOCAL, "global_config", {}).get("assume_finite"))
        """),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    m = importlib.import_module(name)
    try:
        f = c.cache(m.finite)
        with sklearn.config_context(assume_finite=False):
            f(1)
            _reused(c, f)
            scan = hit_memo._Scan(name)
            scan.value(m.LOCAL, frozenset())
            assert scan.probes  # watched, not taken as fixed
            reused = state_memo.STATS["reused"]
            with sklearn.config_context(assume_finite=True):
                got, fresh = _state(c, f)
                assert got == fresh
                assert state_memo.STATS["reused"] == reused  # read again, not served
    finally:
        sys.modules.pop(name, None)


def test_a_watched_setting_that_moved_is_not_served(monkeypatch):
    """`_Family.holds`, which every reuse and every fast hit served in
    Python asks, reads each probe again."""
    box = {"v": 1}
    family = hit_memo._Family((), False, ((lambda: box["v"], 1),))
    state = [None] * 11
    assert family.holds(state)
    box["v"] = 2
    assert not family.holds(state)

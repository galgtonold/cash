"""A call that picks code by a run-time name is cached on a trace of what it ran.

``getattr(module, name)(x)`` and ``importlib.import_module(name)`` choose the
code a call runs from a value, which a key built from the code the function
names cannot see. On Python 3.12+ the call is keyed as any other, its body is
recorded on ``sys.monitoring``, and a later call is served only while every
function that ran, and what each lookup found, is unchanged
(`cash.decorator.code_trace`). Before 3.12 such a call runs uncached
(KEY-UNTRACKABLE-DEP). ``eval``/``exec`` always run uncached.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import warnings

import pytest

from tests._scripts import run_python

TRACED = sys.version_info >= (3, 12)
needs_monitoring = pytest.mark.skipif(not TRACED, reason="sys.monitoring is Python 3.12+")

_PLUGINS = """
def double(x):
    return x * 2


def triple(x):
    return x * 3


def default(x):
    return -x


class Base:
    def score(self, x):
        return x + 100


class Model(Base):
    pass
"""


@pytest.fixture
def load(tmp_path, request):
    """Write ``plugins.py`` and a module of cached functions *body*, import
    both, and count how often a cached body ran (``RUNS``). The module is
    named after the test: an entry an earlier test stored in this process
    is not this one's."""
    module = "m_" + "".join(c if c.isalnum() else "_" for c in request.node.name)
    names = ("plugins", module)
    sys.path.insert(0, str(tmp_path))

    def _load(body: str, plugins: str = _PLUGINS):
        for name in names:
            sys.modules.pop(name, None)
        (tmp_path / "plugins.py").write_text(textwrap.dedent(plugins), encoding="utf-8")
        (tmp_path / f"{module}.py").write_text(
            "import cash, importlib, plugins\nRUNS = []\n" + textwrap.dedent(body), encoding="utf-8"
        )
        importlib.invalidate_caches()
        return importlib.import_module(module)

    yield _load
    sys.path.remove(str(tmp_path))
    for name in names:
        sys.modules.pop(name, None)


_BY_NAME = """
@cash.cache
def pick(name, x):
    RUNS.append(name)  # @cash:assume-safe
    return getattr(plugins, name)(x)
"""


@needs_monitoring
def test_a_second_call_is_served(load):
    m = load(_BY_NAME)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert m.pick("double", 5) == 10
        assert m.pick("double", 5) == 10
    assert m.RUNS == ["double"]
    assert not [w for w in caught if "KEY-UNTRACKABLE-DEP" in str(w.message)]


@needs_monitoring
def test_another_name_runs_another_function(load):
    m = load(_BY_NAME)
    assert m.pick("double", 5) == 10
    assert m.pick("triple", 5) == 15
    assert m.pick("double", 5) == 10
    assert m.RUNS == ["double", "triple"]


@needs_monitoring
def test_a_name_read_from_a_global_that_changes_runs_the_new_function(load):
    m = load(
        """
        NAME = "double"
        @cash.cache
        def pick(x):
            RUNS.append(NAME)  # @cash:assume-safe
            return getattr(plugins, NAME)(x)
        """
    )
    assert m.pick(5) == 10
    m.NAME = "triple"
    assert m.pick(5) == 15
    m.NAME = "double"
    assert m.pick(5) == 10
    assert m.RUNS == ["double", "triple"]


@needs_monitoring
def test_an_edit_to_the_picked_function_runs_it_again(load, tmp_path):
    m = load(_BY_NAME)
    assert m.pick("double", 5) == 10
    (tmp_path / "plugins.py").write_text(
        textwrap.dedent(_PLUGINS).replace("return x * 2", "return x * 20"), encoding="utf-8"
    )
    importlib.reload(m.plugins)
    assert m.pick("double", 5) == 100
    assert m.pick("double", 5) == 100
    assert m.RUNS == ["double", "double"]


@needs_monitoring
def test_an_edit_to_a_function_it_did_not_pick_keeps_the_entry(load, tmp_path):
    m = load(_BY_NAME)
    assert m.pick("double", 5) == 10
    (tmp_path / "plugins.py").write_text(
        textwrap.dedent(_PLUGINS).replace("return x * 3", "return x * 30"), encoding="utf-8"
    )
    importlib.reload(m.plugins)
    assert m.pick("double", 5) == 10
    assert m.RUNS == ["double"]


@needs_monitoring
def test_a_monkeypatched_function_runs_and_the_real_one_comes_back(load, monkeypatch):
    m = load(_BY_NAME)
    assert m.pick("double", 5) == 10
    with monkeypatch.context() as mp:
        mp.setattr(m.plugins, "double", lambda x: 0)
        assert m.pick("double", 5) == 0
    assert m.pick("double", 5) == 10
    assert m.RUNS == ["double", "double", "double"]


@needs_monitoring
def test_an_alias_rebound_to_another_function_is_seen(load):
    m = load(_BY_NAME)
    m.plugins.alias = m.plugins.double
    assert m.pick("alias", 5) == 10
    m.plugins.alias = m.plugins.triple
    assert m.pick("alias", 5) == 15
    assert m.RUNS == ["alias", "alias"]


@needs_monitoring
def test_a_new_module_attribute_that_shadows_the_fallback_is_seen(load):
    """``getattr(mod, name, default)``: the name is not read (three
    arguments), so the whole namespace is checked, and the attribute added
    later is found."""
    m = load(
        """
        @cash.cache
        def pick(name, x):
            RUNS.append(name)  # @cash:assume-safe
            return getattr(plugins, name, plugins.default)(x)
        """
    )
    assert m.pick("fast", 5) == -5
    assert m.pick("fast", 5) == -5
    m.plugins.fast = m.plugins.triple
    assert m.pick("fast", 5) == 15
    assert m.RUNS == ["fast", "fast"]


@needs_monitoring
def test_a_method_a_subclass_adds_over_the_base_one_is_seen(load):
    m = load(
        """
        @cash.cache
        def pick(name, x):
            RUNS.append(name)  # @cash:assume-safe
            return getattr(plugins.Model(), name)(x)
        """
    )
    assert m.pick("score", 1) == 101
    assert m.pick("score", 1) == 101
    m.plugins.Model.score = lambda self, x: x - 100
    assert m.pick("score", 1) == -99
    assert m.RUNS == ["score", "score"]


@needs_monitoring
def test_a_module_imported_by_name_is_checked_by_its_code(load, tmp_path):
    m = load(
        """
        @cash.cache
        def run(module, x):
            RUNS.append(module)  # @cash:assume-safe
            return importlib.import_module(module).double(x)
        """
    )
    assert m.run("plugins", 4) == 8
    assert m.run("plugins", 4) == 8
    (tmp_path / "plugins.py").write_text(
        textwrap.dedent(_PLUGINS).replace("return x * 2", "return x * 2 + 1"), encoding="utf-8"
    )
    importlib.reload(m.plugins)
    assert m.run("plugins", 4) == 9
    assert m.RUNS == ["plugins", "plugins"]


@needs_monitoring
def test_a_picked_function_with_a_side_effect_is_not_served(load):
    m = load(
        _BY_NAME,
        plugins=_PLUGINS + "\n\ndef noisy(x):\n    print('hello')\n    return x\n",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert m.pick("noisy", 5) == 5
        assert m.pick("noisy", 5) == 5
    assert m.RUNS == ["noisy", "noisy"]
    texts = [str(w.message) for w in caught if "KEY-UNTRACKABLE-DEP" in str(w.message)]
    assert texts and "noisy" in texts[0], texts


@pytest.mark.parametrize(
    "name,line", [("by_eval", 'eval("x + 1")'), ("by_exec", 'exec("y = x + 1", {"x": x}, ns := {}) or ns["y"]')]
)
def test_eval_and_exec_still_run_uncached(load, name, line):
    # A name of its own: the warning is shown once per function and process.
    m = load(
        f"""
        @cash.cache
        def {name}(x):
            RUNS.append(x)  # @cash:assume-safe
            return {line}
        """
    )
    compute = getattr(m, name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert compute(1) == 2
        assert compute(1) == 2
    assert m.RUNS == [1, 1]
    assert any("KEY-UNTRACKABLE-DEP" in str(w.message) for w in caught)


@pytest.mark.skipif(TRACED, reason="before 3.12 there is no sys.monitoring to record with")
def test_before_3_12_a_lookup_by_name_runs_uncached(load):
    m = load(_BY_NAME)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert m.pick("double", 5) == 10
        assert m.pick("double", 5) == 10
    assert m.RUNS == ["double", "double"]
    assert any("KEY-UNTRACKABLE-DEP" in str(w.message) for w in caught)


@needs_monitoring
def test_a_new_process_is_served_and_sees_an_edit(tmp_path):
    (tmp_path / "plugins.py").write_text(textwrap.dedent(_PLUGINS), encoding="utf-8")
    (tmp_path / "main.py").write_text(
        textwrap.dedent(
            """
            import importlib, cash

            @cash.cache
            def run(module, name, x):
                print("ran")
                return getattr(importlib.import_module(module), name)(x)

            print(run("plugins", "double", 4))
            """
        ),
        encoding="utf-8",
    )
    first = run_python("main.py", cwd=tmp_path).stdout.split()
    second = run_python("main.py", cwd=tmp_path).stdout.split()
    assert first == ["ran", "8"]
    assert second == ["8"]
    (tmp_path / "plugins.py").write_text(
        textwrap.dedent(_PLUGINS).replace("return x * 2", "return x * 200"), encoding="utf-8"
    )
    assert run_python("main.py", cwd=tmp_path).stdout.split() == ["ran", "800"]


@needs_monitoring
def test_code_the_picked_function_runs_on_a_thread_is_in_the_trace(load, tmp_path):
    pool = """
from concurrent.futures import ThreadPoolExecutor


def part(x):
    return x + 1


def fan_out(x):
    with ThreadPoolExecutor(2) as ex:
        return sum(ex.map(part, [x, x]))
"""
    m = load(_BY_NAME, plugins=pool)
    assert m.pick("fan_out", 1) == 4
    assert m.pick("fan_out", 1) == 4
    (tmp_path / "plugins.py").write_text(pool.replace("return x + 1", "return x + 10"), encoding="utf-8")
    importlib.reload(m.plugins)
    assert m.pick("fan_out", 1) == 22
    assert m.RUNS == ["fan_out", "fan_out"]


@needs_monitoring
def test_a_cached_function_served_inside_stands_for_its_state(load, tmp_path):
    """The picked function is cached itself and served inside the call: its
    body does not run, so the trace names its state, which an edit moves."""
    cached = "import cash\n\n\n@cash.cache\ndef double(x):\n    return x * 2\n"
    m = load(_BY_NAME, plugins=cached)
    assert m.plugins.double(5) == 10  # stored before the call that picks it
    assert m.pick("double", 5) == 10
    assert m.pick("double", 5) == 10
    (tmp_path / "plugins.py").write_text(cached.replace("x * 2", "x * 2 + 25"), encoding="utf-8")
    importlib.reload(m.plugins)
    assert m.pick("double", 5) == 35
    assert m.RUNS == ["double", "double"]

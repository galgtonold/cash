"""``with cash.assume_safe():`` -- the line waiver, for every line of a block.

``# @cash:assume-safe`` waives one line and ``assume_safe=True`` the whole
function. The block sits between them, and unlike the comment it is code: an
editor completes it, and a misspelling fails on the first run instead of
waiving nothing in silence.

What it waives is what the comment waives, on each line inside it, plus the
effects the first call is SEEN to perform while the block is open, a helper's
included. What it must not do: waive for a function of the user's own that is
merely called ``assume_safe``, reach past its own call, or move the key.

Each waiver test has its control in the same module: the same lines without
the block still warn, so a pass cannot come from warnings being off.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import textwrap
import threading
import warnings
from pathlib import Path

import pytest

import cash
from cash import Cash
from cash.effect_observer import EffectObserver
from cash.exceptions import CashImpureFunctionError
from cash.source_norm import source_identity_digest


def _load(tmp_path: Path, source: str):
    """Import *source* as a module from a real file, with ``c`` a fresh Cash.

    A file, not a function in this one: the waiver is resolved against the
    module's own names, which is what a user's module gives it.
    """
    name = f"assume_safe_block_{abs(hash((str(tmp_path), source))) % 10**9}"
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    module.tmp = tmp_path
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def load(tmp_path):
    loaded = []

    def _do(source):
        module = _load(tmp_path, source)
        loaded.append(module.__name__)
        return module

    yield _do
    for name in loaded:
        sys.modules.pop(name, None)


def _codes(call) -> list[str]:
    """The warning codes *call* raises."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        call()
    return [getattr(w.message, "code", None) for w in record if getattr(w.message, "code", None)]


SIDE_EFFECT = """
import os
import cash

@c.cache
def waived(x):
    with cash.assume_safe():
        os.remove("/nonexistent/cash-test") if x < 0 else None
    return x

@c.cache
def plain(x):
    os.remove("/nonexistent/cash-test") if x < 0 else None
    return x

@c.cache
def partly(x):
    with cash.assume_safe():
        os.remove("/nonexistent/cash-test") if x < 0 else None
    os.remove("/nonexistent/cash-test-2") if x < 0 else None
    return x
"""


def test_a_side_effect_inside_the_block_is_waived(load):
    m = load(SIDE_EFFECT)
    assert "IMPURE-SIDE-EFFECTS" in _codes(lambda: m.plain(1)), "control"
    assert _codes(lambda: m.waived(1)) == []


def test_a_line_after_the_block_is_reported(load):
    """The block covers its own lines, as the comment covers its own line."""
    m = load(SIDE_EFFECT)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        m.partly(1)
    [warning] = [w for w in record if getattr(w.message, "code", None) == "IMPURE-SIDE-EFFECTS"]
    lines = Path(m.__file__).read_text(encoding="utf-8").splitlines()
    after = next(i for i, text in enumerate(lines, 1) if "cash-test-2" in text)
    assert 'cash-test")' in lines[after - 2], "the line above it is the waived one"
    assert f"line {after}:" in str(warning.message)
    assert f"line {after - 1}:" not in str(warning.message)


def test_strict_honours_the_block_as_it_honours_the_comment(load):
    m = load(
        """
        import os
        import cash

        @c.cache(strict=True)
        def waived(x):
            with cash.assume_safe():
                os.remove("/nonexistent/cash-test") if x < 0 else None
            return x

        @c.cache(strict=True)
        def plain(x):
            os.remove("/nonexistent/cash-test") if x < 0 else None
            return x
        """
    )
    with pytest.raises(CashImpureFunctionError):
        m.plain(1)
    assert m.waived(1) == 1


GLOBAL_READ = """
import threading
import cash

LOCK = threading.Lock()

@c.cache
def waived(x):
    with cash.assume_safe():
        state = LOCK.locked()
    return (x, state)

@c.cache
def plain(x):
    state = LOCK.locked()
    return (x, state)
"""


def test_a_global_read_inside_the_block_is_waived(load):
    m = load(GLOBAL_READ)
    assert "KEY-UNHASHABLE-GLOBAL" in _codes(lambda: m.plain(1)), "control"
    assert _codes(lambda: m.waived(1)) == []


def test_an_ambient_read_inside_the_block_is_waived(load):
    m = load(
        """
        from datetime import datetime
        import cash

        @c.cache
        def waived(x):
            with cash.assume_safe():
                t = datetime.now()
            return x, t.year

        @c.cache
        def plain(x):
            t = datetime.now()
            return x, t.year
        """
    )
    assert "KEY-AMBIENT-READ" in _codes(lambda: m.plain(1)), "control"
    assert _codes(lambda: m.waived(1)) == []


def test_a_waived_call_on_a_global_does_not_key_on_it(load):
    """What the ledger moves on every call is not an input, inside the block
    as on a commented line: keying on it would make every call a miss."""
    m = load(
        """
        import os
        import cash

        class Ledger:
            def __init__(self):
                self.n = 0

            def record(self, r):
                self.n += 1  # @cash:assume-safe

        LEDGER = Ledger()

        @c.cache
        def waived(x):
            os.write(1, b"")
            with cash.assume_safe():
                LEDGER.record(x)
            return x * 2
        """
    )
    m.waived(1)
    m.waived(1)
    assert m.waived.cache_info()["hits"] == 1


OBSERVED = """
import zipfile
import cash

def archive(target):
    with zipfile.ZipFile(target, "w") as z:
        return z.namelist()

@c.cache
def waived(name):
    with cash.assume_safe():
        with zipfile.ZipFile(str(tmp / name), "w") as z:
            names = z.namelist()
    return str(names)

@c.cache
def plain(name):
    with zipfile.ZipFile(str(tmp / name), "w") as z:
        names = z.namelist()
    return str(names)

@c.cache
def through_a_helper(name):
    with cash.assume_safe():
        names = archive(str(tmp / name))
    return str(names)

@c.cache
def helper_outside(name):
    names = archive(str(tmp / name))
    return str(names)
"""


def test_an_observed_effect_inside_the_block_is_waived(load):
    m = load(OBSERVED)
    assert "IMPURE-OBSERVED-EFFECTS" in _codes(lambda: m.plain("a.zip")), "control"
    assert _codes(lambda: m.waived("b.zip")) == []


def test_an_effect_in_a_helper_called_inside_the_block_is_waived(load):
    """The comment needs a line of the user's on the way to the effect; the
    block waives whatever runs while it is open, however deep."""
    m = load(OBSERVED)
    assert "IMPURE-OBSERVED-EFFECTS" in _codes(lambda: m.helper_outside("a.zip")), "control"
    assert _codes(lambda: m.through_a_helper("b.zip")) == []


def test_a_helpers_own_static_finding_is_waived_in_the_helper_only(load):
    """As with the comment: the block covers the lines in it. A finding in a
    helper's source is waived in that source, so it holds for every caller."""
    m = load(
        """
        import os
        import cash

        def remove(x):
            os.remove("/nonexistent/cash-test") if x < 0 else None

        @c.cache
        def caller(x):
            with cash.assume_safe():
                remove(x)
            return x
        """
    )
    assert "IMPURE-SIDE-EFFECTS" in _codes(lambda: m.caller(1))


def test_a_code_argument_line_inside_the_block_is_waived(load):
    m = load(
        """
        import math
        import cash

        class Waived:
            def run(self, name):
                with cash.assume_safe():
                    return getattr(math, name)(1.0)

        class Plain:
            def run(self, name):
                return getattr(math, name)(1.0)

        @c.cache
        def f(obj):
            return obj.run("sqrt")
        """
    )
    assert "KEY-DYNAMIC-DEPENDENCY" in _codes(lambda: m.f(m.Plain())), "control"
    assert "KEY-DYNAMIC-DEPENDENCY" not in _codes(lambda: m.f(m.Waived()))


@pytest.mark.parametrize(
    ("imports", "spelling"),
    [
        ("import cash", "cash.assume_safe()"),
        ("import cash as c2", "c2.assume_safe()"),
        ("from cash import assume_safe", "assume_safe()"),
        ("from cash import assume_safe as waive", "waive()"),
    ],
    ids=["module", "module-alias", "name", "name-alias"],
)
def test_every_way_of_importing_it_waives(load, imports, spelling):
    m = load(
        f"""
        import os
        {imports}

        @c.cache
        def waived(x):
            with {spelling}:
                os.remove("/nonexistent/cash-test") if x < 0 else None
            return x
        """
    )
    assert _codes(lambda: m.waived(1)) == []


def test_an_import_inside_the_function_waives(load):
    m = load(
        """
        import os

        @c.cache
        def waived(x):
            from cash import assume_safe

            with assume_safe():
                os.remove("/nonexistent/cash-test") if x < 0 else None
            return x
        """
    )
    assert _codes(lambda: m.waived(1)) == []


def test_a_function_of_the_users_named_assume_safe_does_not_waive(load):
    m = load(
        """
        import contextlib
        import os
        import types

        import cash

        def assume_safe():
            return contextlib.nullcontext()

        helpers = types.SimpleNamespace(assume_safe=assume_safe)

        @c.cache
        def own(x):
            with assume_safe():
                os.remove("/nonexistent/cash-test") if x < 0 else None
            return x

        @c.cache
        def attribute(x):
            with helpers.assume_safe():
                os.remove("/nonexistent/cash-test") if x < 0 else None
            return x

        @c.cache
        def cashs(x):
            with cash.assume_safe():
                os.remove("/nonexistent/cash-test") if x < 0 else None
            return x
        """
    )
    assert "IMPURE-SIDE-EFFECTS" in _codes(lambda: m.own(1))
    assert "IMPURE-SIDE-EFFECTS" in _codes(lambda: m.attribute(1))
    assert _codes(lambda: m.cashs(1)) == [], "cash's own, in the same module, waives"


def test_nested_blocks_keep_the_outer_waiver(load, tmp_path):
    """Leaving the inner block restores the outer one, not "no waiver"."""
    m = load(
        """
        import zipfile
        import cash

        def archive(name):
            with zipfile.ZipFile(str(tmp / name), "w") as z:
                return z.namelist()

        @c.cache
        def nested(x):
            with cash.assume_safe():
                with cash.assume_safe():
                    archive("inner.zip")
                archive("between.zip")
            return x

        @c.cache
        def after(x):
            with cash.assume_safe():
                with cash.assume_safe():
                    archive("inner2.zip")
            archive("after.zip")
            return x
        """
    )
    assert _codes(lambda: m.nested(1)) == []
    [message] = [str(w.message) for w in _warnings(lambda: m.after(1)) if w.message.code == "IMPURE-OBSERVED-EFFECTS"]
    assert "after.zip" in message and "inner2.zip" not in message


def _warnings(call):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        call()
    return [w for w in record if getattr(w.message, "code", None)]


def _observed(call) -> list[str]:
    """The IMPURE-OBSERVED-EFFECTS messages *call* raises. The gate and the
    barrier the concurrent tests share are findings of their own."""
    return [str(w.message) for w in _warnings(call) if w.message.code == "IMPURE-OBSERVED-EFFECTS"]


def test_a_cached_call_inside_the_block_is_not_waived(load):
    """The block is about the function it is written in. A cached function
    it calls is judged on its own: its first call has an observer of its own."""
    m = load(
        """
        import zipfile
        import cash

        @c.cache
        def inner(name):
            with zipfile.ZipFile(str(tmp / name), "w") as z:
                return str(z.namelist())

        @c.cache
        def outer(name):
            with cash.assume_safe():
                return inner(name)
        """
    )
    [message] = [str(w.message) for w in _warnings(lambda: m.outer("x.zip"))]
    assert "inner" in message


def test_it_works_in_an_async_function(load):
    """Two calls at once on one loop: the one inside the block is waived,
    the other is not, although the block is open while the other runs."""
    m = load(
        """
        import asyncio
        import zipfile
        import cash

        @c.cache
        async def waived(name):
            with cash.assume_safe():
                await asyncio.sleep(0.2)
                with zipfile.ZipFile(str(tmp / name), "w") as z:
                    names = z.namelist()
            return str(names)

        @c.cache
        async def plain(name):
            await asyncio.sleep(0)
            with zipfile.ZipFile(str(tmp / name), "w") as z:
                names = z.namelist()
            return str(names)
        """
    )

    async def both():
        await asyncio.gather(m.waived("a.zip"), m.plain("b.zip"))

    warned = _observed(lambda: asyncio.run(both()))
    assert len(warned) == 1 and "plain" in warned[0] and "b.zip" in warned[0], warned


def test_it_works_in_a_generator(load):
    m = load(
        """
        import zipfile
        import cash

        @c.cache
        def waived(name):
            with cash.assume_safe():
                with zipfile.ZipFile(str(tmp / name), "w") as z:
                    yield z.namelist()
            yield 2

        @c.cache
        def plain(name):
            with zipfile.ZipFile(str(tmp / name), "w") as z:
                yield z.namelist()
            yield 2
        """
    )
    assert "IMPURE-OBSERVED-EFFECTS" in _codes(lambda: list(m.plain("a.zip"))), "control"
    assert _codes(lambda: list(m.waived("b.zip"))) == []


def test_two_threads_one_inside_a_block(load):
    """Each thread has its own waiver: the block open in one thread must not
    quiet the effect the other performs meanwhile."""
    m = load(
        """
        import threading
        import zipfile
        import cash

        BARRIER = threading.Barrier(2, timeout=10)

        @c.cache
        def waived(name):
            with cash.assume_safe():
                BARRIER.wait()
                with zipfile.ZipFile(str(tmp / name), "w") as z:
                    names = z.namelist()
                BARRIER.wait()
            return str(names)

        @c.cache
        def plain(name):
            BARRIER.wait()
            with zipfile.ZipFile(str(tmp / name), "w") as z:
                names = z.namelist()
            BARRIER.wait()
            return str(names)
        """
    )

    def both():
        threads = [
            threading.Thread(target=m.waived, args=("a.zip",)),
            threading.Thread(target=m.plain, args=("b.zip",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

    warned = _observed(both)
    assert len(warned) == 1 and "plain" in warned[0] and "b.zip" in warned[0], warned


def test_outside_a_cached_call_it_does_nothing(tmp_path):
    from cash.effect_observer import waived_observer

    target = tmp_path / "out.txt"
    with cash.assume_safe() as value:
        target.write_text("x", encoding="utf-8")
        assert waived_observer.get() is None
    assert value is None and target.read_text(encoding="utf-8") == "x"


def test_it_waives_only_the_observer_that_was_open_when_it_began(tmp_path):
    """An observer opened inside the block, for a cached call made there, is
    not waived; the outer one is again once that call returns."""
    outer, inner = EffectObserver(), EffectObserver()
    with outer, cash.assume_safe():
        outer.record_effect("file write", "outer")
        with inner:
            inner.record_effect("file write", "inner")
        outer.record_effect("file write", "outer again")
    assert outer.effects == [] and [d for _, d in inner.effects] == ["inner"]


def test_it_is_exported_and_a_misspelling_fails_loudly():
    from cash import assume_safe

    assert assume_safe is cash.assume_safe and "assume_safe" in cash.__all__
    with pytest.raises(AttributeError):
        cash.asume_safe()  # type: ignore[attr-defined]


# ------------------------------------------------------------ the cache key
BARE = """
def f(rows):
    total = sum(rows)
    log(total)
    save(total)
    return total
"""

WRAPPED = """
def f(rows):
    total = sum(rows)
    with cash.assume_safe():
        log(total)
        save(total)
    return total
"""


@pytest.mark.parametrize(
    "wrapped",
    [
        WRAPPED,
        WRAPPED.replace("cash.assume_safe()", "assume_safe()"),
        WRAPPED.replace("cash.assume_safe()", "c2.assume_safe()"),
        WRAPPED.replace(
            "    with cash.assume_safe():\n", "    with cash.assume_safe():\n        with cash.assume_safe():\n"
        )
        .replace("        log", "            log")
        .replace("        save", "            save"),
    ],
    ids=["cash", "name", "alias", "nested"],
)
def test_the_wrapper_is_not_part_of_the_key(wrapped):
    assert source_identity_digest(wrapped) == source_identity_digest(BARE)


def test_a_code_change_inside_the_block_still_moves_the_key():
    edited = WRAPPED.replace("save(total)", "save(total + 1)")
    assert source_identity_digest(edited) != source_identity_digest(WRAPPED) == source_identity_digest(BARE)
    assert source_identity_digest(edited) == source_identity_digest(BARE.replace("save(total)", "save(total + 1)"))


def test_another_item_on_the_with_line_stays_in_the_key():
    """``with cash.assume_safe(), open(p) as fh:`` is ``with open(p) as fh:``."""
    both = "def f(p):\n    with cash.assume_safe(), open(p) as fh:\n        return fh.read()\n"
    alone = "def f(p):\n    with open(p) as fh:\n        return fh.read()\n"
    bare = "def f(p):\n    return open(p).read()\n"
    assert source_identity_digest(both) == source_identity_digest(alone) != source_identity_digest(bare)


def test_adding_the_block_keeps_the_stored_result(tmp_path):
    """End to end: the same call, before and after the lines are wrapped,
    is a hit, and an edit inside the block is a miss."""
    work = tmp_path / "work"
    work.mkdir()
    path = work / "wrapmod.py"
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    def load(source):
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        spec = importlib.util.spec_from_file_location("wrapmod", path)
        module = importlib.util.module_from_spec(spec)
        module.c = c
        sys.modules["wrapmod"] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop("wrapmod", None)
        return module

    bare = """
        import cash

        @c.cache
        def f(x):
            y = x * 2
            print(y)
            return y
        """
    wrapped = bare.replace("            print(y)\n", "            with cash.assume_safe():\n                print(y)\n")
    edited = wrapped.replace("print(y)", "print(y + 1)")
    assert wrapped != bare

    first = load(bare)
    first.f(1)
    key = first.f.explain(1).cache_key
    second = load(wrapped)
    assert second.f.explain(1).cache_key == key
    assert second.f.explain(1).would_hit
    third = load(edited)
    assert third.f.explain(1).cache_key != key

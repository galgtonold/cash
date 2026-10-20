"""What ``isinstance`` against an ABC, or ``cls.__subclasses__()``,
answers is keyed, though no namespace holds it.

``Shape.register(Circle)`` changes ``isinstance(Circle(), Shape)``, and
``class P(Plugin)`` changes ``Plugin.__subclasses__()``, with no write to a
dict a watcher or the key reads: the call was served the old answer. The key
now names the classes an ABC registered and the classes below it, and the
classes below any class whose reader asks for ``__subclasses__``.
"""

from __future__ import annotations

import abc
import textwrap

from tests._scripts import run_python

RUNS = {"n": 0}


class Shape(abc.ABC):
    pass


class Circle:
    pass


class Plugin:
    pass


class Plain:
    pass


def is_shape(x):
    RUNS["n"] += 1
    return (x, isinstance(Circle(), Shape))


def plugin_names(x):
    RUNS["n"] += 1
    return (x, sorted(k.__name__ for k in Plugin.__subclasses__()))


def plain_name(x):
    RUNS["n"] += 1
    return (x, Plain.__name__)


def test_a_registration_reruns_the_call(cash_instance):
    f = cash_instance.cache(is_shape)
    assert f(1) == f(1) == (1, False)
    runs = RUNS["n"]
    Shape.register(Circle)
    assert f(1) == (1, True)
    assert RUNS["n"] == runs + 1
    assert f(1) == (1, True)
    assert RUNS["n"] == runs + 1


def test_a_new_subclass_reruns_a_call_that_lists_them(cash_instance):
    f = cash_instance.cache(plugin_names)
    assert f(1) == f(1) == (1, [])
    runs = RUNS["n"]

    class Csv(Plugin):
        pass

    assert f(1) == (1, ["Csv"])
    assert RUNS["n"] == runs + 1
    assert f(1) == (1, ["Csv"])
    assert RUNS["n"] == runs + 1


def test_a_new_subclass_keeps_a_call_that_does_not_list_them(cash_instance):
    f = cash_instance.cache(plain_name)
    f(1)
    runs = RUNS["n"]

    class Below(Plain):
        pass

    assert f(1) == (1, "Plain")
    assert RUNS["n"] == runs


SCRIPT = textwrap.dedent("""
    import abc
    import cash

    class Shape(abc.ABC):
        pass

    class Circle:
        pass

    Shape.register(Circle)

    class Plugin:
        pass

    class Csv(Plugin):
        pass

    @cash.cache
    def answers(x):
        print("ran")
        return isinstance(Circle(), Shape), [k.__name__ for k in Plugin.__subclasses__()]

    print(answers(1))
""")


def test_the_key_is_the_same_in_the_next_process(tmp_path):
    script = tmp_path / "family.py"
    script.write_text(SCRIPT, encoding="utf-8")
    first = run_python(script, cwd=tmp_path, cache_dir=tmp_path / ".cash")
    second = run_python(script, cwd=tmp_path, cache_dir=tmp_path / ".cash")
    assert first.stdout.split("\n")[:2] == ["ran", "(True, ['Csv'])"], first.stderr
    assert second.stdout.strip() == "(True, ['Csv'])", second.stderr

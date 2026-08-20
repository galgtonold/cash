"""Identity for code whose source cannot be read.

When ``inspect.getsource`` fails -- ``exec``-defined code, a REPL, a
source file that moved -- the only description of what a callable does is
its compiled form. Two properties matter and pull against each other:

* it must SEE changes, including ones only visible in ``co_consts``.
  ``co_code`` alone is blind to them, because the operand is an index
  into the const table, not the value.
* it must be STABLE across processes. A nested code object's ``repr``
  embeds a memory address, so folding consts in naively produces a
  different digest on every run and nothing ever hits again.
"""

import textwrap

from cash.source_norm import bytecode_identity


def _fn(body, name="f"):
    ns = {}
    exec(compile(textwrap.dedent(body), "<probe>", "exec"), ns)
    return ns[name]


# --- must SEE changes --------------------------------------------------

def test_string_constant_change_is_seen():
    """`co_code` is identical here -- this is the case it cannot see."""
    a = _fn('def f():\n    return "alpha"')
    b = _fn('def f():\n    return "omega"')
    assert a.__code__.co_code == b.__code__.co_code, "premise: co_code identical"
    assert bytecode_identity(a) != bytecode_identity(b)


def test_large_int_constant_change_is_seen():
    a = _fn("def f():\n    return 100000")
    b = _fn("def f():\n    return 200000")
    assert a.__code__.co_code == b.__code__.co_code, "premise: co_code identical"
    assert bytecode_identity(a) != bytecode_identity(b)


def test_body_change_is_seen():
    a = _fn("def f(x):\n    return x + 1")
    b = _fn("def f(x):\n    return x * 2")
    assert bytecode_identity(a) != bytecode_identity(b)


def test_change_inside_a_nested_function_is_seen():
    """The reason consts are RECURSED into rather than dropped."""
    a = _fn(
        """
        def f():
            def inner():
                return "alpha"
            return inner()
        """
    )
    b = _fn(
        """
        def f():
            def inner():
                return "omega"
            return inner()
        """
    )
    assert bytecode_identity(a) != bytecode_identity(b)


def test_change_inside_a_lambda_const_is_seen():
    a = _fn('def f():\n    return (lambda: "alpha")()')
    b = _fn('def f():\n    return (lambda: "omega")()')
    assert bytecode_identity(a) != bytecode_identity(b)


# --- must be STABLE ----------------------------------------------------

def test_identical_source_gives_identical_digest():
    src = "def f(x):\n    return x + 1"
    assert bytecode_identity(_fn(src)) == bytecode_identity(_fn(src))


def test_nested_code_does_not_leak_an_address():
    """Two separate compiles put the nested code object at different
    addresses. If its ``repr`` reached the digest, these would differ --
    and every process would miss the cache forever."""
    src = """
        def f():
            def inner():
                return 1
            return inner()
        """
    a, b = _fn(src), _fn(src)
    assert a.__code__ is not b.__code__, "premise: distinct code objects"
    assert bytecode_identity(a) == bytecode_identity(b)


def test_deeply_nested_code_is_stable():
    src = """
        def f():
            def a():
                def b():
                    def c():
                        return 1
                    return c()
                return b()
            return a()
        """
    assert bytecode_identity(_fn(src)) == bytecode_identity(_fn(src))


# --- shapes without a plain __code__ -----------------------------------

def test_callable_instance_uses_its_call():
    ns = {}
    exec(
        compile(
            textwrap.dedent(
                '''
                class C:
                    def __call__(self):
                        return "alpha"
                '''
            ),
            "<probe>",
            "exec",
        ),
        ns,
    )
    inst = ns["C"]()
    assert bytecode_identity(inst) is not None


def test_no_code_at_all_returns_none():
    """A C builtin has neither source nor bytecode; say so rather than lie."""
    assert bytecode_identity(len) is None


def test_never_raises_on_odd_input():
    for odd in (None, 42, "text", object()):
        bytecode_identity(odd)

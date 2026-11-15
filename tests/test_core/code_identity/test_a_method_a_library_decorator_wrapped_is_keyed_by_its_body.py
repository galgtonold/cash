"""A method a decorator wrapped with ``functools.wraps`` (sklearn's
``@_fit_context``) is the method's code too, not the wrapper's alone.

`cash.loaded_code.code_objects_of` gave a class's wrapped method as the
wrapper's code only, while a function's ``__wrapped__`` chain was followed:
what a memo of the class checks against was blind to the method below. It
now follows the chain for a class's methods as well. Edited on disk, the
method's body still changes the key of a call it runs in.
"""

from __future__ import annotations

import functools
import json
import textwrap

from cash.loaded_code import code_objects_of
from tests._scripts import run_python


def _checked(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return method(self, *args, **kwargs)

    return wrapper


class _Square:
    def __init__(self, side):
        self.side = side

    @_checked
    def size(self):
        return self.side * self.side


def test_a_wrapped_method_gives_the_wrapper_code_and_the_method_code():
    wrapper = vars(_Square)["size"]
    codes = code_objects_of(_Square)
    assert wrapper.__code__ in codes
    assert wrapper.__wrapped__.__code__ in codes


def test_without_following_wrapped_only_the_wrapper_code_is_given():
    wrapper = vars(_Square)["size"]
    codes = code_objects_of(_Square, follow_wrapped=False)
    assert wrapper.__code__ in codes
    assert wrapper.__wrapped__.__code__ not in codes


_SHAPES = """
import functools


def checked(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return method(self, *args, **kwargs)

    return wrapper


class Square:
    def __init__(self, side):
        self.side = side

    @checked
    def size(self):
        return self.side * FACTOR
"""

_MAIN = """
import json

import cash
import shapes

c = cash.Cash(register_magic=False)
RUNS = []


@c.cache
def area(shape):
    RUNS.append(1)  # @cash:assume-safe
    return shape.size()


print(json.dumps({"area": area(shapes.Square(3)), "runs": len(RUNS)}))
"""


def test_editing_the_wrapped_method_body_invalidates(tmp_path):
    shapes = tmp_path / "shapes.py"
    shapes.write_text(textwrap.dedent(_SHAPES).replace("FACTOR", "self.side"), encoding="utf-8")
    (tmp_path / "main.py").write_text(textwrap.dedent(_MAIN), encoding="utf-8")

    def run() -> dict:
        return json.loads(run_python("main.py", cwd=tmp_path).stdout.strip().splitlines()[-1])

    assert run() == {"area": 9, "runs": 1}
    assert run() == {"area": 9, "runs": 0}
    shapes.write_text(textwrap.dedent(_SHAPES).replace("FACTOR", "10"), encoding="utf-8")
    assert run() == {"area": 30, "runs": 1}

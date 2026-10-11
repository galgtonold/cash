"""What a loop's helpers do when called, read from their bodies.

A loop run as one unit names what it changed by its key only when its outcome
is a function of that key. The check passed each helper's ``def`` to the
statement scan, which skips function bodies on purpose (defining a function
reads no clock), so ``def stamp(i): return time.time()`` was never seen, and
the list the loop filled kept its old lineage with new values. These pin the
two readers that check now uses.
"""

import textwrap

import pytest

from cash.analysis.code_analyzer import CodeAnalyzer
from cash.tracking.randomness import mints_unseeded_generator


@pytest.mark.parametrize(
    "src, reason",
    [
        ("import time\ndef stamp(i):\n    return time.time()", "time.time"),
        ("import uuid\ndef tag(i):\n    def inner():\n        return uuid.uuid4()\n    return inner()", "uuid.uuid4"),
        ("import os\nf = lambda i: os.urandom(1)", "os.urandom"),
        ("def pick(i):\n    import secrets\n    return secrets.randbelow(9)", "secrets.randbelow"),
    ],
    ids=["a body", "a nested function", "a lambda", "an import in the body"],
)
def test_a_function_body_is_scanned(src, reason):
    assert reason in CodeAnalyzer.scan_function_bodies_for_forbidden_functions(src, {})
    assert reason not in CodeAnalyzer.scan_for_forbidden_functions(src, {}), "the statement scan still skips bodies"


def test_a_clean_body_finds_nothing():
    src = "import math\ndef f(i):\n    return math.sqrt(i) * 2"
    assert CodeAnalyzer.scan_function_bodies_for_forbidden_functions(src, {}) == []


@pytest.mark.parametrize(
    "src",
    [
        "def f():\n    return np.random.default_rng().integers(9)",
        "g = numpy.random.default_rng(None)",
        "g = default_rng()",
        "g = np.random.RandomState()",
        "g = random.Random()",
        "g = random.SystemRandom(0)",
        "g = np.random.Generator(np.random.PCG64())",
    ],
)
def test_a_generator_seeded_from_entropy_is_found(src):
    assert mints_unseeded_generator(src)


@pytest.mark.parametrize(
    "src",
    [
        "def f():\n    return np.random.default_rng(0).integers(9)",
        "g = np.random.default_rng(seed)",
        "g = random.Random(42)",
        "g = np.random.Generator(np.random.PCG64(1))",
        "g = rng.integers(9)",
    ],
)
def test_a_seeded_generator_is_not(src):
    assert not mints_unseeded_generator(textwrap.dedent(src))

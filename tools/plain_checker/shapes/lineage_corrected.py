"""rnl triage: group shapes from starter.py plus corrected adaptations."""

from tools.plain_checker import notebook

GROUP = {
    "call-by-string-runs-once",
    "module-csv-chunks",
    "module-deque-popleft",
    "module-dict-iter-then-pop",
    "module-generator-next",
    "module-generator-send",
    "module-heappop",
    "module-helper-calls-next",
    "module-object-draw-method",
    "draw-by-string-in-helper-cell",
    "draw-twice-in-one-cell",
    "module-edit-back-to-equal-result",
    "module-seeded-rng-instance",
    "draw-twice-in-try-blocks",
    "draw-twice-in-with-blocks",
}

_PRE = "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\n"


def _hs(src):
    return (
        _PRE
        + "SRC_HS_SAME = "
        + repr(src)
        + "\nif not os.path.exists('hs_same.py') or open('hs_same.py').read() != SRC_HS_SAME:\n    with open('hs_same.py', 'w') as fh:\n        fh.write(SRC_HS_SAME)\nimport importlib, hs_same\nimportlib.reload(hs_same)"
    )


SRC_LOG = (
    "import os\n"
    "def gen():\n"
    "    i = 0\n"
    "    while True:\n"
    "        fd = os.open('hfl_calls.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
    "        os.write(fd, b'x')\n"
    "        os.close(fd)\n"
    "        yield i\n"
    "        i += 1\n"
    "G = gen()\n"
)

SRC_CNT = (
    "import os\n"
    "def bump():\n"
    "    fd = os.open('hdw_calls.bin', os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
    "    os.write(fd, b'x')\n"
    "    os.close(fd)\n"
)

EXTRA = [
    notebook(
        "draw-by-string-in-helper-cell-logged",
        cells=[
            _PRE
            + "SRC = "
            + repr(SRC_LOG)
            + "\nif not os.path.exists('hfl_bydraw.py') or open('hfl_bydraw.py').read() != SRC:\n    with open('hfl_bydraw.py', 'w') as fh:\n        fh.write(SRC)\nfrom hfl_bydraw import G",
            "name = 'draw'",
            "def draw():\n    return next(G)",
            "t = globals()[name]()",
            "u = t + 1",
        ],
        names=["t", "u"],
        source="reviews/rerun-new-lineage-2/review.md F3 (f_byname_draw), with the os.write log per draw kept",
    ),
    notebook(
        "call-by-string-runs-once-oswrite",
        cells=[
            _PRE
            + "SRC = "
            + repr(SRC_CNT)
            + "\nif not os.path.exists('hdw_count.py') or open('hdw_count.py').read() != SRC:\n    with open('hdw_count.py', 'w') as fh:\n        fh.write(SRC)\nimport hdw_count",
            "name = 'make'",
            "def make(x):\n    hdw_count.bump()\n    return x + 1",
            "t = globals()[name](3)",
            "u = t + 0",
            "w = u * 2",
            "n = os.stat('hdw_calls.bin').st_size  # @cash:no-cache",
        ],
        names=["t", "u", "w", "n"],
        source="reviews/rerun-new-lineage/review.md F1 Repro A (d_byname_count), counted with os.write (cumulative, like the review)",
    ),
    notebook(
        "module-edit-back-to-equal-result-restart",
        cells=[
            _hs("def f(x, k=2):\n    return x * k\n"),
            "def g(x):\n    import hs_same\n    return hs_same.f(x)",
            "t = g(3)",
            "u = t + 1",
        ],
        names=["t", "u"],
        edits={
            2: {0: _hs("def f(x, k=10):\n    return x * k\n")},
            3: {0: _hs("def f(x, k=5):\n    return x * k * 2\n")},
        },
        source="reviews/rerun-new-lineage-2/review.md F1 Repro B (s_samevalue_mod): k=10 -> 5*2 change made across the restart",
    ),
]

SHAPES = EXTRA

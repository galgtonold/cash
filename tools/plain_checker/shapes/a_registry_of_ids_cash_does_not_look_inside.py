"""Values keeping id()s where the walk does not look: a registry of more
items than it reads, a pandas Series indexed by ids, and round numbers in a
big notebook that are no ids (from the batch 16 review)."""

from tools.plain_checker import notebook

_P = "class P:\n    pass\n"
SHAPES = [
    notebook(
        "o-id-dict-5k",
        [
            _P + "ps = [P() for _ in range(5000)]",
            "reg = {id(p): i for i, p in enumerate(ps)}",
            "hit = reg.get(id(ps[0]))",
        ],
        names=["hit"],
    ),
    notebook(
        "o-id-dict-20k",
        [
            _P + "ps = [P() for _ in range(20000)]",
            "reg = {id(p): i for i, p in enumerate(ps)}",
            "hit = reg.get(id(ps[0]))",
        ],
        names=["hit"],
    ),
    notebook(
        "o-str-id",
        [
            _P + "ps = [P() for _ in range(10)]",
            "reg = {str(id(p)): i for i, p in enumerate(ps)}",
            "hit = reg.get(str(id(ps[0])))",
        ],
        names=["hit"],
    ),
    notebook(
        "o-id-series",
        [
            _P + "import pandas as pd\nps = [P() for _ in range(10)]",
            "reg = pd.Series(range(10), index=[id(p) for p in ps])",
            "hit = reg.get(id(ps[0]))",
        ],
        names=["hit"],
    ),
    notebook(
        "o-int-big-ns",
        [
            "data = [[{'k': i} for i in range(9000)] for _ in range(25)]",
            "import time\nx = (time.sleep(0.5), 1 << 20)[1]",
            "y = x + 1",
        ],
        names=["x", "y"],
    ),
    notebook(
        "o-int-small-ns",
        [
            "data = [[{'k': i} for i in range(90)] for _ in range(25)]",
            "import time\nx = (time.sleep(0.5), 1 << 20)[1]",
            "y = x + 1",
        ],
        names=["x", "y"],
    ),
]

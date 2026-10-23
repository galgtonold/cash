"""Corrected / extra shapes for triage group e4r."""

from tools.plain_checker import notebook

SHAPES = [
    notebook(
        "e4r-namespace-memo-cell",
        cells=[
            "import functools, time, types\ndef slow(n):\n    time.sleep(0.02); return n * n\ndef mk_ns():\n    return types.SimpleNamespace(f=functools.cache(slow))",
            "sizes = []\nfor k in range(2):\n    h = mk_ns()\n    sizes.append(h.f.cache_info().currsize)\n    h.f(1); h.f(2)",
        ],
        names=["sizes"],
        source="reviews/made-memo-not-served/review.md F1 cell-level repro (memo in a SimpleNamespace)",
    ),
    notebook(
        "e4r-reducer-module-list-via-reduce",
        cells=[
            "import os, sys\nsys.dont_write_bytecode = True\nif os.getcwd() not in sys.path:\n    sys.path.insert(0, os.getcwd())\nSRC_RMOD = 'import time\\ndef work(tag):\\n    time.sleep(0.25)\\n    return tag\\nSEEN = []\\nclass D:\\n    def __reduce__(self):\\n        SEEN.append(1)\\n        return (D, ())\\n'\nif not os.path.exists('rmod_e4r.py') or open('rmod_e4r.py').read() != SRC_RMOD:\n    with open('rmod_e4r.py', 'w') as fh:\n        fh.write(SRC_RMOD)\nimport rmod_e4r as rmod\nfrom rmod_e4r import work",
            "d = (work('d'), rmod.D())[1]",
            "k = len(rmod.SEEN)",
        ],
        names=["k"],
        source="reviews/e-r5-reducer-r2/review.md R4 (mod_attr_read, faithful: __reduce__ + slow work())",
    ),
]

SHAPES.append(
    notebook(
        "e4r-file-rewritten-closed-writes",
        cells=[
            "import os\nP = 'f.txt'",
            "with open(P, 'w') as fh:\n    fh.write('1')\nfor _ in range(1):\n    t = int(open(P).read())\nu = t\nwith open(P, 'w') as fh:\n    fh.write('2')",
            "w = u",
        ],
        names=["t", "u", "w"],
        source="diagnostic variant of file-rewritten-later-in-cell: writes closed by with",
    )
)

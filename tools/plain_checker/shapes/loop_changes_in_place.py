"""One-unit loops that change an object in place."""

from tools.plain_checker import notebook

P = "import time\nclass P:\n    def __init__(self):\n        self.n = 0\n        self.seen = []\n    def update(self, i=0):\n        self.n += 1\n        self.seen.append(i)\n"

SHAPES = [
    notebook(
        "loop-update-alias-name",
        cells=[
            P + "p = P()\nq = p",
            "for i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)",
            "r = (p.n, q.n, p is q, len(q.seen))",
        ],
        names=["r"],
    ),
    notebook(
        "loop-update-alias-container",
        cells=[
            P + "p = P()\nholder = {'a': [p]}",
            "for i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)",
            "r = (holder['a'][0].n, holder['a'][0] is p)",
        ],
        names=["r"],
    ),
    notebook(
        "loop-update-alias-made-after",
        cells=[
            P + "p = P()",
            "for i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)",
            "q = p\nr = (q.n, q is p)",
        ],
        names=["r"],
    ),
    notebook(
        "loop-update-read-later",
        cells=[
            P + "p = P()",
            "for i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)",
            "s = p.n * 2 + sum(p.seen)",
            "t = s + 1",
        ],
        names=["s", "t"],
    ),
    notebook(
        "loop-update-some-passes",
        cells=[
            P + "p = P()",
            "for i in range(3000):\n    time.sleep(0.0003)\n    if i % 7 == 0:\n        p.update(i)",
            "r = (p.n, p.seen[-1])",
        ],
        names=["r"],
    ),
    notebook(
        "loop-update-same-cell",
        cells=[P, "p = P()\nfor i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)\nr = p.n"],
        names=["r"],
    ),
    notebook(
        "loop-update-range-extended",
        cells=[
            P + "p = P()",
            "for i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)",
            "r = (p.n, sum(p.seen))",
        ],
        names=["r"],
        edits={
            2: {1: "for i in range(3500):\n    time.sleep(0.0003)\n    p.update(i)"},
            3: {1: "for i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)"},
        },
    ),
    notebook(
        "loop-update-two-objects-and-list",
        cells=[
            P + "p = P()\nw = P()\nout = []",
            "for i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)\n    out.append(i)\n    if i % 2:\n        w.update(-i)",
            "r = (p.n, w.n, len(out), sum(w.seen))",
        ],
        names=["r"],
    ),
    notebook(
        "loop-update-dict-of-counts",
        cells=[
            "import time\nfrom collections import Counter\nc = Counter()\nd = {}",
            "for i in range(3000):\n    time.sleep(0.0003)\n    c.update([i % 5])\n    d.setdefault(i % 3, []).append(i)",
            "r = (sorted(c.items()), {k: len(v) for k, v in d.items()})",
        ],
        names=["r"],
    ),
    notebook(
        "loop-update-holds-stdout",
        cells=[
            "import sys\n" + P + "p = P()\np.stream = sys.stdout",
            "for i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)",
            "r = (p.n, p.stream is sys.stdout)",
        ],
        names=["r"],
    ),
    notebook(
        "loop-update-while",
        cells=[
            P + "p = P()",
            "i = 0\nwhile i < 3000:\n    time.sleep(0.0003)\n    p.update(i)\n    i += 1",
            "r = (p.n, i)",
        ],
        names=["r"],
    ),
]

SHAPES += [
    notebook(
        "loop-update-closure-holds",
        cells=[
            P + "p = P()\ndef mk(o):\n    return lambda: (o.n, len(o.seen))\nget = mk(p)",
            "for i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)",
            "r = get()",
        ],
        names=["r"],
    ),
    notebook(
        "loop-update-attribute-holds",
        cells=[
            P + "class Box:\n    pass\nbox = Box()\np = P()\nbox.obj = p",
            "for i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)",
            "r = (box.obj.n, box.obj is p)",
        ],
        names=["r"],
    ),
    notebook(
        "loop-update-earlier-cached-value-holds",
        cells=[
            P + "def slow_pair(o):\n    time.sleep(0.05)\n    return [o, 1]\np = P()",
            "pair = slow_pair(p)",
            "for i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)",
            "r = (pair[0].n, pair[0] is p)",
        ],
        names=["r"],
    ),
    notebook(
        "loop-update-default-arg-holds",
        cells=[
            P + "p = P()\ndef look(o=p):\n    return o.n",
            "for i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)",
            "r = (look(), p.n)",
        ],
        names=["r"],
    ),
    notebook(
        "loop-update-item-of-list",
        cells=[
            P + "ps = [P(), P()]\np = ps[0]",
            "for i in range(3000):\n    time.sleep(0.0003)\n    p.update(i)",
            "r = (ps[0].n, ps[0] is p, ps[1].n)",
        ],
        names=["r"],
    ),
    notebook(
        "loop-update-through-alias-in-loop",
        cells=[
            P + "p = P()\nq = p",
            "for i in range(3000):\n    time.sleep(0.0003)\n    q.update(i)",
            "r = (p.n, q.n, p is q)",
        ],
        names=["r"],
    ),
]

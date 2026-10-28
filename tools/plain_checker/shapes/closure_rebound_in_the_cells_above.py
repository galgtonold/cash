"""A closure the cells above point elsewhere by a call that binds nothing.

``draw, rebind = make(BIG)`` then ``rebind([0])``: ``draw`` reads ``[0]``.
After ``BIG.append`` below, the reader re-run alone was rebuilt without
``rebind([0])`` and read all of ``BIG``, in the processes whose string hashes
linked ``draw`` (not ``rebind``) to ``BIG``.
"""

from tools.plain_checker import notebook

_SETUP = (
    "import time\nBIG = [{'id': i} for i in range(20000)]\n"
    "def make(rows):\n    def draw(k):\n        time.sleep(0.05)\n        return len(rows)\n"
    "    def rebind(r):\n        nonlocal rows\n        rows = r\n    return draw, rebind\n"
)

SHAPES = [
    notebook(
        "closure-rebound-reader-alone",
        [_SETUP + "draw, rebind = make(BIG)\nrebind([0])", "y = draw(1)\nprint(y)", "BIG.append({'id': -1})"],
        names=["y"],
        plan=["all", 1, 1, 2, 1, "restart", "all", 2, 1],
    ),
    notebook(
        "closure-rebound-split-reader-alone",
        [_SETUP, "draw, rebind = make(BIG)", "rebind([0])", "y = draw(1)\nprint(y)", "BIG.append({'id': -1})"],
        names=["y"],
        plan=["all", 3, 3, 4, 3, "restart", "all", 4, 3],
    ),
]

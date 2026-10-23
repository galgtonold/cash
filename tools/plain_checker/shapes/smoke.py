"""Three quick shapes that check the checker itself."""

from tools.plain_checker import decorator, notebook

SHAPES = [
    notebook(
        "smoke-list-append",
        cells=[
            "import time\nxs = [1, 2, 3]",
            "time.sleep(0.05)\ntotal = sum(xs)\nprint('total', total)",
            "xs.append(4)",
            "time.sleep(0.05)\nagain = sum(xs)\nprint('again', again)",
        ],
        edits={3: {0: "import time\nxs = [10, 20]"}},
        source="checker smoke test",
    ),
    notebook(
        "smoke-identical-cells",
        cells=["xs = [1]", "xs.append(len(xs))\nprint(xs)", "xs.append(len(xs))\nprint(xs)"],
        source="checker smoke test: each cell goes with its id, so two that read the same are told apart",
    ),
    decorator(
        "smoke-decorator-mutation",
        setup="import time\n@cache\ndef total(xs):\n    time.sleep(0.05)\n    return sum(xs)\n",
        steps=["xs = [1, 2]", "total(xs)", "xs.append(3)", "total(xs)"],
        source="checker smoke test",
    ),
]

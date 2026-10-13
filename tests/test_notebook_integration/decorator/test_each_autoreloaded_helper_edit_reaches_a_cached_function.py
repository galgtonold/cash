"""``%autoreload 2`` over a run of edits: every edit of a helper, of the
cached function, or of both, is what the next call returns.

IPython 9's ``%autoreload 2`` (deduperreload) recompiles each edited
function from its definition alone and swaps the code into the old object:
the code says it starts on line 1 of the module (a decorated function's,
of the decorator's file), so reading its source at that line gave
``import cash``. The cached function's own text then held no call to the
helper, the key dropped the helper, and every later helper edit was served
the old result (104/104/4/405 where a plain kernel gives
204/304/305/505). Both the decorator alone and ``%cash_on`` are run.
"""

import os
import time
import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

TEMPLATE = (
    "import cash\n"
    "def _ratio(a, b):\n"
    "    return a / b + {h}\n"
    "@cash.cache\n"
    "def feat(x):\n"
    "    return _ratio(x, 2) + {f}\n"
    "@cash.cache\n"
    "def build(x):\n"
    "    return feat(x) * 10\n"
)

#: (helper constant, feat constant) per edit, in the order of the edit sequence that went stale.
EDITS = [(0, 2), (100, 2), (200, 2), (300, 2), (300, 3), (400, 3), (500, 3)]


@pytest.mark.parametrize("with_cash", [False, True], ids=["decorator", "cash_on"])
def test_every_edit_is_the_value_a_plain_kernel_gives(nb_runner, with_cash):
    name = f"arseq_{uuid.uuid4().hex[:8]}"
    path = nb_runner.work_dir / f"{name}.py"
    # Distinct mtimes in the past: autoreload sees each edit, and each file
    # version has settled (cash memoises what it reads from a settled file).
    stamp = time.time() - 600

    def write(h: int, f: int) -> None:
        nonlocal stamp
        path.write_text(TEMPLATE.format(h=h, f=f), encoding="utf-8", newline="\n")
        stamp += 10
        os.utime(path, (stamp, stamp))

    write(0, 1)
    nb_runner.create_notebook(
        [
            f"%load_ext autoreload\n%autoreload 2\nimport {name} as m",
            "print(m.feat(4), m.build(4))",
        ]
    )
    nb_runner.start_kernel(with_cash=with_cash)
    nb_runner.run_all()
    assert nb_runner.get_output(2).split() == ["3.0", "30.0"]

    for h, f in EDITS:
        write(h, f)
        want = [f"{2.0 + h + f}", f"{(2.0 + h + f) * 10}"]
        for _ in range(2):
            nb_runner.run_cell(2)
            assert nb_runner.get_output(2).split() == want, (h, f, nb_runner.get_raw_output(2))

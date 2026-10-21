"""Review repros: the user's object inside a library object a call returns,
compared with plain Python across two runs and a restart.

The statement call unit only: a ``@cash.cache`` function hands back its
stored result, so ``w.rows is rect`` is False there by design."""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

SETUP = (
    "import cash\n%cash_on\nimport time, argparse, dataclasses, collections\nimport numpy as np, pandas as pd\n"
    "import matplotlib\nmatplotlib.use('Agg')\nfrom matplotlib.figure import Figure\n"
    "from sklearn.neighbors import KNeighborsRegressor"
)
SLEEP = "    time.sleep(1.0)\n"
L = "rect = [1]\n"
CASES = {
    # name: (data cell, call cell, change, read)
    "dict_in_namespace": (
        L + "def mk():\n" + SLEEP + "    return argparse.Namespace(d={'r': rect})",
        "w = mk()",
        "rect.append(5)",
        "print('SAME', w.d['r'] is rect, w.d['r'][-1])",
    ),
    "dataclass_in_namespace": (
        L
        + "@dataclasses.dataclass\nclass B:\n    r: list\ndef mk():\n"
        + SLEEP
        + "    return argparse.Namespace(b=B(rect))",
        "w = mk()",
        "rect.append(5)",
        "print('SAME', w.b.r is rect, w.b.r[-1])",
    ),
    "namedtuple_in_namespace": (
        L + "P = collections.namedtuple('P', 'r')\ndef mk():\n" + SLEEP + "    return argparse.Namespace(p=P(rect))",
        "w = mk()",
        "rect.append(5)",
        "print('SAME', w.p.r is rect, w.p.r[-1])",
    ),
    "dataframe_attrs": (
        L + "def mk():\n" + SLEEP + "    df = pd.DataFrame({'a': [1]})\n    df.attrs['r'] = rect\n    return df",
        "w = mk()",
        "rect.append(5)",
        "print('SAME', w.attrs['r'] is rect, w.attrs['r'][-1])",
    ),
    "knn_fit_on_global": (
        "rect = np.random.default_rng(0).random((20, 2))\ny = np.arange(20.0)\ndef mk():\n"
        + SLEEP
        + "    return KNeighborsRegressor(algorithm='brute', n_neighbors=2).fit(rect, y)",
        "w = mk()",
        "rect[0, 0] = 99.0",
        "print('SAME', w._fit_X is rect, w._fit_X[0, 0])",
    ),
    "knn_fit_on_arg": (
        "rect = np.random.default_rng(0).random((20, 2))\ny = np.arange(20.0)\ndef mk(X):\n"
        + SLEEP
        + "    return KNeighborsRegressor(algorithm='brute', n_neighbors=2).fit(X, y)",
        "w = mk(rect)",
        "rect[0, 0] = 99.0",
        "print('SAME', w._fit_X is rect, w._fit_X[0, 0])",
    ),
    "line_data_in_namespace": (
        "rect = np.arange(3.0)\ndef mk():\n"
        + SLEEP
        + "    fig = Figure()\n    fig.add_subplot().plot(rect)\n    return argparse.Namespace(fig=fig)",
        "w = mk()",
        "rect[0] = 9.0",
        "d = w.fig.axes[0].lines[0].get_ydata(orig=True)\nprint('SAME', d is rect, d[0])",
    ),
    "namespace_of_arg": (
        L + "def mk(r):\n" + SLEEP + "    return argparse.Namespace(rows=r)",
        "w = mk(rect)",
        "rect.append(5)",
        "print('SAME', w.rows is rect, w.rows[-1])",
    ),
    "call_plain_dict_arg": (
        L + "def mk(r):\n" + SLEEP + "    return {'rows': r}",
        "w = mk(rect)",
        "rect.append(5)",
        "print('SAME', w['rows'] is rect, w['rows'][-1])",
    ),
    "dict_in_namespace_arg": (
        L + "def mk(r):\n" + SLEEP + "    return argparse.Namespace(d={'r': [r]})",
        "w = mk(rect)",
        "rect.append(5)",
        "print('SAME', w.d['r'][0] is rect, w.d['r'][0][-1])",
    ),
    "user_obj_in_dataclass_in_ns_arg": (
        L
        + "@dataclasses.dataclass\nclass B:\n    r: list\ndef mk(r):\n"
        + SLEEP
        + "    return argparse.Namespace(b=[B(r)])",
        "w = mk(rect)",
        "rect.append(5)",
        "print('SAME', w.b[0].r is rect, w.b[0].r[-1])",
    ),
    "part_of_a_library_arg": (
        L + "src = argparse.Namespace(rows=rect)\ndef mk(s):\n" + SLEEP + "    return argparse.Namespace(rows=s.rows)",
        "w = mk(src)",
        "rect.append(5)",
        "print('SAME', w.rows is rect, w.rows[-1])",
    ),
    "part_of_a_library_global": (
        L + "src = argparse.Namespace(rows=rect)\ndef mk():\n" + SLEEP + "    return argparse.Namespace(rows=src.rows)",
        "w = mk()",
        "rect.append(5)",
        "print('SAME', w.rows is rect, w.rows[-1])",
    ),
    "part_of_a_fig_arg": (
        "rect = None\nfig = Figure()\nfig.add_subplot()\ndef mk(f):\n"
        + SLEEP
        + "    return argparse.Namespace(ax=f.axes[0])",
        "w = mk(fig)",
        "fig.axes[0].set_title('t')",
        "print('SAME', w.ax is fig.axes[0], 5)",
    ),
}


def _same_line(text):
    return next((line for line in text.splitlines() if line.startswith("SAME")), text)


@pytest.mark.parametrize("case", sorted(CASES))
def test_like_plain(nb_runner, case):
    data, call, change, read = CASES[case]
    cells = [SETUP, data, call, change, read]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    seen = []
    for restart in (False, False, True):
        if restart:
            nb_runner.restart()
        nb_runner.run_all()
        seen.append(_same_line(nb_runner.get_raw_output(5)))
    print(case, "cash:", seen)
    plain = (
        "SAME False 0.0"
        if case == "line_data_in_namespace"
        else "SAME True "
        + {
            "rect.append(5)": "5",
            "rect[0, 0] = 99.0": "99.0",
            "rect[0] = 9.0": "9.0",
            "fig.axes[0].set_title('t')": "5",
        }[change]
        if case != "line_data_in_namespace"
        else ""
    )
    assert all(s.strip() == plain for s in seen), (case, plain, seen)

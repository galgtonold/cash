"""The same call, judged by both paths.

Each row is one call, written as ``r = <call>`` in a notebook statement and as
``r = <call>; return r`` in a ``@cash.cache`` function, with the verdict each
path gives it. The two paths used to keep their own lists of effect names,
which drifted apart; the rows below are the calls where they disagreed, and
what each path does about them now that both read ``cash.effects``.

Notebook verdicts: ``refuse`` (the statement runs every time) or ``cache``.
Decorator verdicts: the purity issue kinds the analyzer reports, or
``silent``.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import textwrap

import pytest

from cash.analysis.annotations import CacheAnnotation, get_statement_annotations
from cash.analysis.cacheability import analyze_statement
from cash.analysis.cacheability_decision import decide_cacheability
from cash.analysis.code_analyzer import CodeAnalyzer
from cash.purity_analyzer import PurityAnalyzer

_HEADER = """\
import csv, datetime, getpass, gzip, json, os, pickle, re, shutil, subprocess, sys, time, uuid
import urllib.request
from pathlib import Path

a = "a"; b = "b"; u = "http://example.invalid"; sql = "insert into t values (1)"
"""

_PARAMS = "p, df, f, cur, conn, session, sock, client, s3"

#: (call, notebook verdict, decorator verdict)
ROWS = [
    # Files and folders: the decorator's list stopped at five `to_*` names,
    # and neither path had `copyfile`, `os.link`, `chmod`, `touch`, `unlink`.
    ("shutil.copytree(a, b)", "refuse", "impure_call"),
    ("shutil.copyfile(a, b)", "refuse", "impure_call"),
    ("os.symlink(a, b)", "refuse", "impure_call"),
    ("os.link(a, b)", "refuse", "impure_call"),
    ("os.chmod(a, 0o644)", "refuse", "impure_call"),
    ("csv.writer(f)", "refuse", "impure_call"),
    ("df.to_hdf(a, key='k')", "refuse", "impure_call"),
    ("df.to_feather(a)", "refuse", "impure_call"),
    ("df.to_html(a)", "refuse", "impure_call"),
    ("p.mkdir()", "refuse", "impure_call"),
    ("p.touch()", "refuse", "impure_call"),
    ("p.unlink()", "refuse", "impure_call"),
    # Other programs: the notebook refused os.system but not os.popen, and
    # the decorator saw os.popen only by watching the first call spawn it.
    ("os.popen('true')", "refuse", "impure_call"),
    ("subprocess.getoutput('true')", "refuse", "impure_call"),
    # The network, named in full: the decorator named requests.get but not
    # requests.head or urlopen; the notebook refused requests.post but not the
    # same POST spelled requests.request("POST", ...) or urlopen(url, data).
    # A read is cached in a notebook, like reading a file, and gets the TTL
    # advisory (KEY-NETWORK-READ) in a decorated function.
    ("requests.head(u)", "cache", "network_read"),
    ("urllib.request.urlopen(u)", "cache", "network_read"),
    ("requests.request('GET', u)", "cache", "network_read"),
    ("requests.request('POST', u)", "refuse", "impure_call"),
    ("urllib.request.urlopen(u, b'x=1')", "refuse", "impure_call"),
    # The same write through a client object: the notebook cached
    # `session.post(u)` while refusing `requests.post(u)`.
    ("session.post(u)", "refuse", "impure_call"),
    ("sock.sendall(b'x')", "refuse", "impure_call"),
    ("client.publish(a, b)", "refuse", "impure_call"),
    ("s3.upload_file(a, b, 'k')", "refuse", "impure_call"),
    ("s3.put_object(Bucket=a, Key=b)", "refuse", "impure_call"),
    # A database: a notebook cache hit skipped `cur.execute("INSERT ...")` and
    # `conn.commit()`, while `df.to_sql` ran every time (as a "file write") and
    # was silent in a decorated function. A literal SELECT is a read.
    ("cur.execute(sql)", "refuse", "impure_call"),
    ("cur.executemany(sql, [])", "refuse", "impure_call"),
    ("conn.commit()", "refuse", "impure_call"),
    ("df.to_sql('t', conn)", "refuse", "impure_call"),
    ("cur.execute('SELECT 1')", "cache", "silent"),
    # The clock: each path kept its own list. The notebook missed
    # `time.strftime("%Y")` and `datetime.today()` and refused
    # `time.localtime(ts)`, which only converts `ts`; the decorator missed
    # `time.process_time()`.
    ("time.strftime('%Y')", "refuse", "ambient_read"),
    ("datetime.datetime.today()", "refuse", "ambient_read"),
    ("time.process_time()", "refuse", "ambient_read"),
    ("time.localtime(1.0)", "cache", "silent"),
    ("time.strftime('%Y', time.localtime(1.0))", "cache", "silent"),
    # pyplot's current figure: a notebook statement drawing on it runs every
    # time, and a decorated function that draws on it was silent.
    ("plt.plot([1, 2])", "refuse", "impure_call"),
    ("plt.show()", "refuse", "impure_call"),
    # The person at the keyboard: `input()` was cached in a notebook (a hit
    # replayed the first answer without asking), and `getpass` was silent in
    # a decorated function.
    ("input()", "refuse", "impure_call"),
    ("getpass.getpass()", "refuse", "impure_call"),
    # A builtin's name on some other receiver is not the builtin: the
    # decorator reported these as "known I/O", the notebook never did.
    ("re.compile('a+')", "cache", "silent"),
    ("gzip.open(a)", "cache", "silent"),
    ("df.eval('x + 1')", "cache", "silent"),
]


@pytest.fixture(scope="module")
def probe_module(tmp_path_factory):
    """One module with a function per row, so the decorator's analyzer reads
    each call from real source."""
    path = tmp_path_factory.mktemp("verdicts") / "_effect_verdicts_probe.py"
    body = _HEADER
    for i, (call, _nb, _dec) in enumerate(ROWS):
        body += f"\n\ndef f{i}({_PARAMS}):\n    r = {call}\n    return r\n"
    path.write_text(body)
    spec = importlib.util.spec_from_file_location("_effect_verdicts_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def notebook_verdict(
    statement: str, namespace: dict, annotation: CacheAnnotation | None = None
) -> tuple[str, list[str]]:
    tree = ast.parse(statement)
    cacheable, reasons = decide_cacheability(
        code=statement,
        tree=tree,
        inputs=set(),
        outputs={"r"},
        annotation=annotation,
        analysis=analyze_statement(statement, tree),
        user_ns=namespace,
        variable_lineage={},
        is_stateful_call=lambda _name: False,
        scan_forbidden=CodeAnalyzer.scan_for_forbidden_functions,
    )
    return ("cache" if cacheable else "refuse"), reasons


def decorator_verdict(func) -> tuple[str, list[str]]:
    issues = PurityAnalyzer().analyze(func).issues
    kinds = sorted({issue.kind for issue in issues})
    return (",".join(kinds) or "silent"), [issue.description for issue in issues]


@pytest.mark.parametrize(("index", "row"), list(enumerate(ROWS)), ids=[row[0] for row in ROWS])
def test_notebook_verdict(probe_module, index, row):
    call, expected, _ = row
    verdict, reasons = notebook_verdict(f"r = {call}", dict(vars(probe_module)))
    assert verdict == expected, reasons


@pytest.mark.parametrize(("index", "row"), list(enumerate(ROWS)), ids=[row[0] for row in ROWS])
def test_decorator_verdict(probe_module, index, row):
    _call, _, expected = row
    verdict, descriptions = decorator_verdict(getattr(probe_module, f"f{index}"))
    assert verdict == expected, descriptions


def test_rows_are_real_calls():
    """A row that does not parse as one call would test nothing."""
    for call, _nb, _dec in ROWS:
        node = ast.parse(textwrap.dedent(call), mode="eval").body
        assert isinstance(node, ast.Call), call


def test_a_database_write_is_named_as_one(probe_module):
    """`df.to_sql` used to be reported as a file write."""
    _, reasons = notebook_verdict("r = df.to_sql('t', conn)", dict(vars(probe_module)))
    assert reasons == ["Side effect: df.to_sql() (database_write)"]


def test_a_pandas_clock_read_is_refused_and_reported(tmp_path):
    """`pd.Timestamp.now()` warned in a decorated function and was cached in a
    notebook statement."""
    pd = pytest.importorskip("pandas")
    path = tmp_path / "_effect_verdicts_pandas.py"
    path.write_text("import pandas as pd\n\n\ndef f():\n    r = pd.Timestamp.now()\n    return r\n")
    spec = importlib.util.spec_from_file_location("_effect_verdicts_pandas", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert decorator_verdict(module.f)[0] == "ambient_read"
    verdict, reasons = notebook_verdict("r = pd.Timestamp.now()", {"pd": pd})
    assert verdict == "refuse" and reasons == ["Timestamp.now"]


def test_a_discarded_pyplot_call_is_reported_once(tmp_path):
    path = tmp_path / "_effect_verdicts_plot.py"
    path.write_text("def f(plt):\n    plt.title('t')\n    return 1\n")
    spec = importlib.util.spec_from_file_location("_effect_verdicts_plot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert decorator_verdict(module.f) == (
        "impure_call",
        ["plt.title() - draws on pyplot's current figure, which a hit does not redraw"],
    )


def test_the_keyboard_is_recognised_through_an_alias():
    import getpass

    assert notebook_verdict("r = ask()", {"ask": getpass.getpass}) == ("refuse", ["getpass.getpass"])
    assert notebook_verdict("r = ask()", {"ask": input}) == ("refuse", ["input"])
    assert notebook_verdict("r = ask()", {"ask": len}) == ("cache", [])


def _annotated_verdict(statement: str, namespace: dict) -> tuple[str, list[str]]:
    """The verdict with the annotation read from the statement's own comment."""
    tree = ast.parse(statement)
    return notebook_verdict(statement, namespace, get_statement_annotations(statement, tree.body[0]))


@pytest.mark.parametrize(
    "call",
    ["session.post(u, json={'q': 1})", "requests.post(u)", "cur.execute(sql)", "shutil.copyfile(a, b)"],
)
def test_assume_safe_overrides_a_side_effect_refusal(probe_module, call):
    """A POST that only runs a query cannot be told from one that writes; the
    user can, and says so on the line."""
    namespace = dict(vars(probe_module))
    assert notebook_verdict(f"r = {call}", namespace)[0] == "refuse"
    assert _annotated_verdict(f"r = {call}  # @cash:assume-safe", namespace) == ("cache", [])
    assert _annotated_verdict(f"# @cash:assume-safe\nr = {call}", namespace) == ("cache", [])


@pytest.mark.parametrize("call", ["time.time()", "input()"])
def test_assume_safe_does_not_cache_a_value_read_each_time(probe_module, call):
    """The clock and the keyboard are not side effects a hit skips: a hit
    would replay the first answer."""
    verdict, _ = _annotated_verdict(f"r = {call}  # @cash:assume-safe", dict(vars(probe_module)))
    assert verdict == "refuse"


def test_assume_safe_does_not_waive_an_in_place_mutation(probe_module):
    namespace = {**vars(probe_module), "rows": []}
    statement = "rows.append(session.post(u))  # @cash:assume-safe"
    tree = ast.parse(statement)
    cacheable, reasons = decide_cacheability(
        code=statement,
        tree=tree,
        inputs=set(),
        outputs=set(),
        annotation=get_statement_annotations(statement, tree.body[0]),
        analysis=analyze_statement(statement, tree),
        user_ns=namespace,
        variable_lineage={},
        is_stateful_call=lambda _name: False,
        scan_forbidden=CodeAnalyzer.scan_for_forbidden_functions,
    )
    assert not cacheable and reasons[0] == "In-place mutation on: rows"
    assert not any(reason.startswith("Side effect") for reason in reasons)


def test_assume_safe_in_a_cell_header_stays_on_its_statement(probe_module):
    """A waiver spread over a whole cell would cache writes nobody audited."""
    cell = "# @cash:assume-safe\n\nr = requests.post(u)\ns = session.post(u)\n"
    tree = ast.parse(cell)
    assert not get_statement_annotations(cell, tree.body[1]).assume_safe

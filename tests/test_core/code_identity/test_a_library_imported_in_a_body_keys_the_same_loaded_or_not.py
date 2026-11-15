"""A library a function body imports keys the call the same whether an
earlier step of the process already loaded it or not: by the name imported
and the library's installed version, never by the object only a loaded
module offers. Keyed by that object, the second call of a process that
loaded it in the first missed ("no longer uses global local:X"), and a run
that loaded it earlier missed what a run that did not had stored."""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.timeout(300)]

JOB = """
import sys, warnings
import cash

warnings.simplefilter("ignore")
if sys.argv[1] == "loaded":
    import {module}  # an earlier step of the program loaded it


@cash.cache
def f(x):
    from {module} import {name}
    return {name}.__name__ + x


print(f("a"), f("a"), f.cache_info()["hits"])
"""

CASES = {"stdlib": ("email.mime.text", "MIMEText"), "installed": ("six", "moves")}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_second_call_of_a_process_hits(tmp_path, case):
    module, name = CASES[case]
    (tmp_path / "job.py").write_text(textwrap.dedent(JOB).format(module=module, name=name), encoding="utf-8")
    out = run_python("job.py", "cold", cwd=tmp_path)
    *values, hits = out.stdout.split()
    assert values[0] == values[1] and hits == "1", out.stdout


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_run_that_loaded_it_first_is_served_what_one_that_did_not_stored(tmp_path, case):
    module, name = CASES[case]
    (tmp_path / "job.py").write_text(textwrap.dedent(JOB).format(module=module, name=name), encoding="utf-8")
    run_python("job.py", "cold", cwd=tmp_path)
    out = run_python("job.py", "loaded", cwd=tmp_path)
    assert out.stdout.split()[-1] == "2", out.stdout


def _fake_parseaddr(text):
    return ("fake", text)


def test_a_library_name_rebound_between_calls_is_keyed_by_what_it_holds(disk_cash, monkeypatch):
    """Loaded, the name is read once: rebound to anything but the library's
    own object (a patch, the user's function), the call is keyed by that,
    and every call returns what plain Python does."""
    import email.utils

    def parse(text):
        from email.utils import parseaddr

        return parseaddr(text)

    cached = disk_cash.cache(parse)
    seen = [cached("a <b@c>")]
    monkeypatch.setattr(email.utils, "parseaddr", _fake_parseaddr)
    seen.append(cached("a <b@c>"))
    monkeypatch.undo()
    seen.append(cached("a <b@c>"))
    plain = [parse("a <b@c>")]
    monkeypatch.setattr(email.utils, "parseaddr", _fake_parseaddr)
    plain.append(parse("a <b@c>"))
    monkeypatch.undo()
    plain.append(parse("a <b@c>"))
    assert seen == plain == [("a", "b@c"), ("fake", "a <b@c>"), ("a", "b@c")]
    assert cached.cache_info()["hits"] == 1

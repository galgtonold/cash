"""Round-18 configuration papercuts: a config mistake is loud, and where a run
looked for its config is visible.

* A misspelled ``[tool.cash]`` key did nothing, silently, while
  ``cash.configure()`` raised on the same typo.
* ``pytest proj/tests`` typed from the directory above cached per user and
  ignored ``proj/pyproject.toml``; ``python -m pytest`` from the same place
  cached in ``./.cash``.
* ``max_cache_size = "2GB"`` was shown back as ``1.9 GB``.
* ``cash info`` left out ``disable = true``, and every other setting's origin.
* A bad value was printed twice, uncoded; a UTF-8 BOM was reported as an
  invalid statement at line 1, column 1.
* ``explain()`` did not say which cache it had read.
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings

import pytest

from cash import Cash
from cash.config import get_config, parse_size

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

needs_toml = pytest.mark.skipif(
    sys.version_info < (3, 11) and not __import__("importlib").util.find_spec("tomli"),
    reason="no TOML parser")


def _codes(record, code):
    return [str(w.message) for w in record if f"[{code}]" in str(w.message)]


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CASH_", "PYTEST_"))}
    env.update(extra)
    return env


@needs_toml
def test_a_misspelled_key_is_named_with_the_setting_it_meant(tmp_path):
    project = tmp_path / "pyproject.toml"
    project.write_text('[tool.cash]\nmax_cache_siz = "2GB"\n'
                       '[[tool.cash.tiers]]\ntype = "memory"\nmax_entrys = 3\n',
                       encoding="utf-8")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cfg = get_config(project_config_path=str(project), user_config_path=None)
    said = _codes(rec, "CONFIG-UNKNOWN-KEY")
    assert any("`max_cache_siz`" in m and "`max_cache_size`" in m for m in said), said
    assert any("tiers[0].max_entrys" in m and "`max_entries`" in m for m in said), said
    assert cfg.max_cache_size is None


@needs_toml
def test_a_pyproject_without_a_cash_table_is_not_cash_s_to_check(tmp_path):
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nname = "p"\n[tool.black]\nline-length = 88\n',
                       encoding="utf-8")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cfg = get_config(project_config_path=str(project), user_config_path=None)
    assert not _codes(rec, "CONFIG-UNKNOWN-KEY")
    assert "project" not in cfg._source, "a file with no cash settings was counted"


@needs_toml
def test_a_bad_value_is_said_once_however_often_the_config_is_resolved(tmp_path):
    project = tmp_path / "pyproject.toml"
    project.write_text('[tool.cash]\ncompress = "yes please"\n', encoding="utf-8")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(3):
            get_config(project_config_path=str(project), user_config_path=None)
    assert len(_codes(rec, "CONFIG-INVALID")) == 1, [str(w.message) for w in rec]


@needs_toml
def test_a_byte_order_mark_is_named(tmp_path):
    project = tmp_path / "pyproject.toml"
    project.write_bytes(b"\xef\xbb\xbf[tool.cash]\ncompress = true\n")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        get_config(project_config_path=str(project), user_config_path=None)
    said = _codes(rec, "CONFIG-INVALID")
    assert said and "byte-order mark" in said[0], [str(w.message) for w in rec]


@pytest.mark.parametrize("written, shown", [
    ("2GB", "2 GB"), ("512MiB", "512 MiB"), ("1.5GiB", "1.5 GiB"), ("64KB", "64 KB"),
])
def test_a_size_is_shown_in_the_unit_it_was_written_in(written, shown):
    from cash.config import format_size
    assert format_size(parse_size(written)) == shown


@needs_toml
def test_cash_info_lists_every_setting_and_where_it_came_from(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cash]\ndisable = true\nmax_cache_size = "2GB"\n', encoding="utf-8")
    out = subprocess.run([sys.executable, "-m", "cash", "info"], cwd=str(tmp_path),
                         env=_clean_env(CASH_SUMMARY="false"),
                         capture_output=True, text=True, timeout=120).stdout
    assert "Disabled:   yes" in out, out
    assert "2 GB (2,000,000,000 bytes)" in out, out
    settings = out.split("Settings", 1)[1]
    assert "disable = True" in settings and "pyproject.toml" in settings, out
    assert "summary = False" in settings and "CASH_SUMMARY" in settings, out
    assert f"{tmp_path / 'pyproject.toml'}  (read)" in out, out


def test_explain_names_the_cache_it_read(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)

    @c.cache(assume_safe=True)
    def f(x):
        return x

    answer = f.explain(1)
    assert answer.cache_dir == os.path.abspath(tmp_path / ".cash")
    assert f"cache_dir: {answer.cache_dir}" in str(answer)


@needs_toml
@pytest.mark.parametrize("workers", [[], ["-n", "2"]], ids=["serial", "xdist"])
def test_pytest_from_above_the_project_uses_the_project_config(tmp_path, workers):
    if workers:
        pytest.importorskip("xdist")
    project = tmp_path / "proj"
    (project / "tests").mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[project]\nname = "p"\n[tool.cash]\ncache_dir = "mycache"\n', encoding="utf-8")
    out = tmp_path / "where.txt"
    (project / "tests" / "test_where.py").write_text(
        "import os\n"
        "from cash.config import get_config\n"
        "def test_where():\n"
        "    with open(os.environ['WHERE'], 'a') as f:\n"
        "        f.write(str(get_config().cache_dir) + '\\n')\n",
        encoding="utf-8")
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "proj/tests", "-q", "-p", "no:cacheprovider", *workers],
        cwd=str(tmp_path), env=_clean_env(WHERE=str(out)),
        capture_output=True, text=True, timeout=240)
    assert run.returncode == 0, run.stdout + run.stderr
    seen = [os.path.normcase(os.path.realpath(p))
            for p in out.read_text(encoding="utf-8").splitlines()]
    assert seen == [os.path.normcase(os.path.realpath(project / "mycache"))], seen

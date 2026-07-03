"""CAS-83: pathlib write methods must be recognized as file-write side
effects, so a write-only cell re-executes instead of being served from cache
(which would silently skip creating the file)."""

import pytest

pytestmark = [pytest.mark.timeout(90), pytest.mark.files]


def _p(path) -> str:
    return str(path).replace("\\", "/")


def test_write_text_cell_recreates_file_on_rerun(nb_runner, tmp_path):
    out = tmp_path / "out.json"
    nb_runner.create_notebook([
        "from pathlib import Path\ncfg = {'x': 1}",
        f"nchars = Path('{_p(out)}').write_text(str(cfg))",
        f"print('content=' + open('{_p(out)}').read())",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert out.exists(), "setup: write cell did not create the file"
    assert "content={'x': 1}" in nb_runner.get_output(3)

    out.unlink()
    nb_runner.run_cell(2)
    assert out.exists(), (
        "Path.write_text cell served from cache without executing: "
        "out.json was NOT recreated"
    )


def test_write_bytes_cell_recreates_file_on_rerun(nb_runner, tmp_path):
    out = tmp_path / "blob.bin"
    nb_runner.create_notebook([
        "from pathlib import Path",
        f"n = Path('{_p(out)}').write_bytes(b'abc')",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_persist()
    nb_runner.run_all()
    assert out.exists(), "setup: write cell did not create the file"

    out.unlink()
    nb_runner.run_cell(2)
    assert out.exists(), (
        "Path.write_bytes cell served from cache without executing: "
        "blob.bin was NOT recreated"
    )

"""A magic called through the shell lists the folder it runs in, as plain does.

Notebooks saved before IPython 9 call ``get_ipython().magic('ls ')``, the
deprecated alias of ``run_line_magic``; code also reaches a runner with
``getattr``. cash cannot see what such a call reads, so it runs every time,
as ``%ls`` does: cached, it showed the first folder's listing after a
``%cd`` and after a new file.

IPython 9 has no ``magic``: the helper module adds it as IPython 8 defines it.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

SHIM = (
    "from IPython.core.interactiveshell import InteractiveShell\n"
    "if not hasattr(InteractiveShell, 'magic'):\n"
    "    def magic(self, arg_s):\n"
    "        name, _, line = arg_s.partition(' ')\n"
    "        return self.run_line_magic(name, line, _stack_depth=2)\n"
    "    InteractiveShell.magic = magic\n"
)


@pytest.mark.parametrize(
    "listing",
    ["get_ipython().magic('ls ')", "getattr(get_ipython(), 'run_line_magic')('ls', '')"],
)
def test_the_listing_follows_cd_and_a_new_file(nb_runner, listing):
    (nb_runner.work_dir / "ipy8_magic.py").write_text(SHIM, encoding="utf-8")
    (nb_runner.work_dir / "sub").mkdir()
    (nb_runner.work_dir / "sub" / "inside.txt").write_text("", encoding="utf-8")
    (nb_runner.work_dir / "top.txt").write_text("", encoding="utf-8")
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\nimport ipy8_magic",
            listing,
            "%cd sub",
            # Not the same text as cell 2, which a run with no cell ids could not tell apart.
            "# in sub\n" + listing,
            "open('new.txt', 'w').close()\n" + listing,
            "%cd ..",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert "top.txt" in nb_runner.get_output(2)
    assert "inside.txt" in nb_runner.get_output(4)
    assert "top.txt" not in nb_runner.get_output(4)
    assert "new.txt" in nb_runner.get_output(5)

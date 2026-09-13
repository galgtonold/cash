"""An upstream edit reaches a cell that writes into the same frame in place.

Round 23 (r23s4, silent WRONG, 3/3). ``docs`` is loaded in cell 1 and cleaned
in cell 2; cell 3 adds a column in place (``docs["n_chars"] = ...``). After
cell 2 was edited -- a filter, or a changed cleaning step -- running cell 3
printed the pre-edit frame: cell 2 was never replayed. A cell 3 that does not
write into ``docs`` saw the edit.
"""
import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.upstream]

LOAD = ("import time\nimport pandas as pd\n"
        "def load_docs(n=300):\n"
        "    time.sleep(0.2)\n"
        "    return pd.DataFrame({'body': ['Bitte PRUEFEN' if i % 20 == 0 else 'Please CHECK' for i in range(n)]})\n"
        "docs = load_docs()")
CLEAN = "docs['text'] = docs['body'].str.lower()"
CLEAN_FILTERED = CLEAN + "\ndocs = docs[~docs['text'].str.contains('bitte')].reset_index(drop=True)"
CLEAN_REPLACED = "docs['text'] = docs['body'].str.lower().str.replace('bitte', 'please')"
INPLACE = ("docs['n_chars'] = docs['text'].str.len()\n"
           "print('rows', len(docs), 'german', int(docs['text'].str.contains('bitte').sum()))")
READ_ONLY = ("n_chars = docs['text'].str.len()\n"
             "print('rows', len(docs), 'german', int(docs['text'].str.contains('bitte').sum()))")


@pytest.mark.parametrize("edit, expected", [
    (CLEAN_FILTERED, "rows 285 german 0"),
    (CLEAN_REPLACED, "rows 300 german 0"),
], ids=["filter", "replace"])
@pytest.mark.parametrize("last", [INPLACE, READ_ONLY], ids=["inplace", "read-only"])
def test_the_edit_is_seen(nb_runner, edit, expected, last):
    nb_runner.create_notebook(["import cash\n%cash_on", LOAD, CLEAN, last])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "rows 300 german 15" in nb_runner.get_output(4)
    nb_runner.set_cell_source(3, edit)
    nb_runner.run_cell(4)
    assert expected in nb_runner.get_output(4), nb_runner.get_output(4)[-400:]

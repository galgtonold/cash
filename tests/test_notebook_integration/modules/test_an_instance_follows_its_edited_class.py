"""An object built from a class in a local module is rebuilt after the class is edited.

A model class lives in a ``.py`` file, a cell builds the instance, a later
cell uses it: the everyday shape of an ML notebook. Editing the class made
cash reload the module but keep the instance built from the old class, so
``m.predict`` returned the pre-edit result and ``isinstance(m, model.Model)``
was False. A top-to-bottom run of the edited code gives the new result and
True, and so must the cell run on its own.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

MODEL = "class Model:\n    def predict(self, v):\n        return v + {n}\n"


def test_the_cell_using_the_instance_sees_the_edited_class(nb_runner, tmp_path):
    mod = tmp_path / "instmodel.py"
    mod.write_text(MODEL.format(n=1), encoding="utf-8")
    nb_runner.create_notebook(
        ["import instmodel", "m = instmodel.Model()", "p = m.predict(10)\nok = isinstance(m, instmodel.Model)"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("(p, ok)") == "(11, True)"

    mod.write_text(MODEL.format(n=100), encoding="utf-8")
    nb_runner.run_cell(3)

    assert nb_runner.peek("(p, ok)") == "(110, True)", nb_runner.get_raw_output(3)

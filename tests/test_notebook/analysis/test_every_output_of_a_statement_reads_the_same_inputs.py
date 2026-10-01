"""Every output of a statement gets its lineage from the inputs it READ.

``M = enc.fit_transform(data)`` writes ``M`` and, by fitting it, ``enc``. The
runtime read the input lineages once per output, after recording the outputs
before it, so when ``enc`` came first ``M`` was built on ``enc``'s new lineage.
Which came first is set iteration order, which string hashing randomises per
process: a kernel recorded ``M`` one way, the upstream simulation (which reads
the inputs once, before the statement) and the next kernel another, and the
statement downstream of ``M`` missed after the first restart.
"""

import pytest

from cash.notebook.cache_key import statement_source_hash
from cash.notebook.lineage_formula import output_lineage

CODE = "M = enc.fit_transform(data)"


class _Scaler:
    def fit_transform(self, x):
        self.mean_ = sum(x) / len(x)
        return [v - self.mean_ for v in x]


def _record(statement_processor, order):
    state = statement_processor.tracking_state
    user_ns = statement_processor.shell.user_ns
    user_ns["data"] = [1.0, 2.0, 3.0]
    user_ns["enc"] = _Scaler()
    state.lineage.record("data", "d" * 64)
    state.lineage.record("enc", "e" * 64)
    exec(CODE, {}, user_ns)
    # An ordered set view, so the test chooses the order instead of the hash seed.
    outputs = dict.fromkeys(order).keys()
    statement_processor.lineage_builder.capture_and_track_variables(
        state, outputs, {"enc", "data"}, CODE, statement_source_hash(CODE), cache_key=""
    )
    return dict(state.variable_lineage), {v: dict(state.executed_input_lineages[v]) for v in order}


@pytest.mark.parametrize("order", [("enc", "M"), ("M", "enc")], ids=["enc-first", "M-first"])
def test_each_output_is_built_on_the_lineages_before_the_statement(statement_processor, order):
    lineage, read = _record(statement_processor, order)
    expected = output_lineage(statement_source_hash(CODE), ["d" * 64, "e" * 64])
    assert lineage["M"] == expected
    assert lineage["enc"] == expected
    assert read["M"] == read["enc"] == {"data": "d" * 64, "enc": "e" * 64}

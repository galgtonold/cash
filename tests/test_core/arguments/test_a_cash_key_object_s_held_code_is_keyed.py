"""An object keyed by ``__cash_key__`` still brings the code it holds into the key.

``__cash_key__`` names what identifies the object's DATA. The code of what it
holds is code like any other: ``Loader(dataset_id, db)`` whose cached method
calls ``self.db.query(...)`` runs ``FakeDB.query``. The attribute walk that
finds that code was skipped for such an object, so editing ``FakeDB.query``
served the old result, though the same edit recomputed without
``__cash_key__``.
"""

from __future__ import annotations

from tests.test_core.code_identity._edited_project import edited_runs

DB = """
    class FakeDB:
        def query(self, dataset_id, version):
            return f"{dataset_id}@{version}"
"""

LOADER = """
    import cash
    from db import FakeDB

    class Loader:
        def __init__(self, dataset_id, db):
            self.dataset_id = dataset_id
            self.db = db

        def __cash_key__(self):
            return self.dataset_id

        @cash.cache
        def load(self, version):
            return self.db.query(self.dataset_id, version)

    @cash.cache
    def load_with(loader, version):
        return loader.db.query(loader.dataset_id, version)

    print(Loader("sales", FakeDB()).load(1))
    print(load_with(Loader("sales", FakeDB()), 2))
"""

EDIT = [("db.py", 'f"{dataset_id}@{version}"', 'f"{dataset_id}#v{version}"')]


def test_editing_a_held_object_s_class_recomputes(tmp_path):
    first, after, uncached = edited_runs(tmp_path, {"main.py": LOADER, "db.py": DB}, EDIT)
    assert first == "sales@1\nsales@2"
    assert after == uncached == "sales#v1\nsales#v2"

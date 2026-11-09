"""``install_paths.clear_caches`` also forgets which classes are the user's.

The size probes ask per class whether it is the user's (`_users_class`),
remembered per class. A test that changes what counts as the user's code
clears the install caches; the class verdicts went on answering as before.
"""

from __future__ import annotations

from cash import install_paths
from cash._user_attrs import _users_class


class Mine:
    pass


def test_clear_caches_drops_the_remembered_verdicts():
    _users_class(Mine)
    assert _users_class.memo
    install_paths.clear_caches()
    assert not _users_class.memo

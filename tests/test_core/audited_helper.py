"""Fixture for test_assume_safe_annotation: helpers in another module.

The waiver is read from the source of the function that HAS the finding. For a
helper, that is this file -- so annotating here waives it for every caller,
which is the semantic the test pins.
"""


def sink(payload):
    return payload


def audited_helper(uid):
    """Its own impurity, waived here."""
    sink({"audited": uid})              # @cash:assume-safe
    return uid


def unaudited_helper(uid):
    """Identical, without the waiver."""
    sink({"not audited": uid})
    return uid

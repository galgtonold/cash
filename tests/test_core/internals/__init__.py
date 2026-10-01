"""Tests that pin how the decorator does its work, not what a user sees.

Each test here reaches into a private part of cash (a memo, a counter, a
spy on a private method) to check that work is done once, is skipped on a
hit, or stays within a bound. A refactor of the internals is expected to
break them; rewrite them with the new internals. A test of behaviour a
user can observe belongs in the feature folders instead.
"""

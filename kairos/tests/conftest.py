"""Shared pieces for every feature: a fresh `context` per scenario, and the
@known_problem tag."""

import pytest


@pytest.fixture
def context() -> dict:
    """What the Given and When steps found, for the Then steps to check."""
    return {}


def pytest_bdd_apply_tag(tag, function):
    """@known_problem marks a scenario that fails on purpose: it describes how
    the model should behave, and it does not yet. When someone fixes the
    problem the scenario starts passing, and pytest reports it so the tag can
    be removed (strict)."""
    if tag == "known_problem":
        marker = pytest.mark.xfail(reason="known problem, see the comment above the scenario", strict=True)
        marker(function)
        return True
    return None

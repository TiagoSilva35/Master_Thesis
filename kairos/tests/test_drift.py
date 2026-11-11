"""Steps for drift.feature."""

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.change import ChangeFit
from src.causal_model.fitting import BayesianFitting, PlainFitting
from src.causal_model.node import Node
from src.causal_model.outcome import OutcomeNode
from src.causal_model.screening import PairScreening
from tests.helpers import world

scenarios("drift.feature")

WORLDS = {1: 0, 3: 2, 5: 4}  # world number -> position in SCENARIOS


@given(parsers.parse("{before:d} people from before and {after:d} from after the drift in world {number:d}"))
def people_before_and_after(context, before, after, number):
    scenario = world(WORLDS[number])
    context["scenario"] = scenario
    context["old"] = scenario.world.sample(before, 11)
    context["new"] = scenario.drifted.sample(after, 12)


def _joint(context, child: str, ridge: float = 1e-3):
    parents = context["scenario"].world.parents()[child]
    node = Node(child, parents, pairs=[], intervals=16, fitting=BayesianFitting(ridge=ridge))
    return ChangeFit().fit(node, context["old"], context["new"])


@when(parsers.parse("the {child} node is fitted jointly on both"))
def joint_fit(context, child):
    context["result"] = _joint(context, child)


@when("the outcome node is fitted jointly on both, with screening")
def joint_outcome(context):
    parents = context["scenario"].world.parents()["Y"]
    node = OutcomeNode("Y", parents, pairs=[], intervals=8, fitting=BayesianFitting(), screening=PairScreening())
    context["result"] = ChangeFit().fit(node, context["old"], context["new"])


@when(parsers.parse("the {child} node is fitted jointly with ridge {first:g} and with ridge {second:g}"))
def joint_two_ridges(context, child, first, second):
    context["results"] = [_joint(context, child, ridge) for ridge in (first, second)]


@when(parsers.parse("I fit the {child} node jointly with a plain fitting"))
def joint_plain(context, child):
    parents = context["scenario"].world.parents()[child]
    node = Node(child, parents, pairs=[], intervals=8, fitting=PlainFitting())
    with pytest.raises(TypeError) as error:
        ChangeFit().fit(node, context["old"], context["new"])
    context["error"] = error.value


@then(parsers.parse("{term} -> {child} is flagged as changed"))
def flagged(context, term, child):
    result = context["result"]
    assert term in result.changed(), f"gains: {result.gains}"


@then(parsers.parse("{term} -> {child} is not flagged"))
def not_flagged(context, term, child):
    result = context["result"]
    assert term not in result.changed(), f"gains: {result.gains}"


@then(parsers.parse("the gains of {term} -> {child} differ by less than {tolerance:g}"))
def gains_independent_of_ridge(context, term, child, tolerance):
    first, second = (result.gains[term] for result in context["results"])
    assert abs(first - second) < tolerance


@then(parsers.parse("the two level gains differ by less than {tolerance:g}"))
def level_gains_independent_of_ridge(context, tolerance):
    first, second = (result.gains["level"] for result in context["results"])
    assert abs(first - second) < tolerance


@then(parsers.parse("the level of {child} is not flagged"))
def level_not_flagged(context, child):
    assert all("level" not in result.changed() for result in context["results"])


@then("it refuses because it needs Bayesian fitting")
def refuses_plain(context):
    assert "BayesianFitting" in str(context["error"])

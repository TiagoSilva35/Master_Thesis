"""Steps for interventional_rows.feature."""

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.discovery import Background, GraphSearch

scenarios("interventional_rows.feature")


@given(parsers.parse("{count:d} observational people where y = 2x plus bell-shaped noise"))
def observational(context, count):
    random = np.random.default_rng(0)
    x = random.normal(0, 1, count)
    context["data"] = {"x": x, "y": 2 * x + random.normal(0, 1, count)}
    context["intervened"] = None


@given(parsers.parse("{observed:d} observational people where y = 2x plus bell-shaped noise, "
                     "and {set_:d} where x was set around 2"))
def with_interventions(context, observed, set_):
    random = np.random.default_rng(0)
    x = np.concatenate([random.normal(0, 1, observed), random.normal(2, 0.5, set_)])
    context["data"] = {"x": x, "y": 2 * x + random.normal(0, 1, observed + set_)}
    marked = np.concatenate([np.zeros(observed, bool), np.ones(set_, bool)])
    context["intervened"] = {"x": marked, "y": np.zeros(observed + set_, bool)}


@when("the graph is searched")
@when("the graph is searched with the rows that set x marked")
def search(context):
    context["search"] = GraphSearch(Background())
    context["found"] = context["search"].run(context["data"], context["intervened"])


@then("x and y are linked, with the direction flagged doubtful")
def doubtful_link(context):
    found = context["found"]
    assert len(found.arrows()) == 1 and set(found.arrows()[0]) == {"x", "y"}
    assert found.doubtful == found.arrows()


@then("the arrow x → y is found and not flagged doubtful")
def directed(context):
    assert context["found"].arrows() == [("x", "y")] and context["found"].doubtful == []


@then("reversing it loses more than the margin")
def reversal_loses(context):
    search = context["search"]
    score = search.score(context["data"], context["intervened"])
    forward = score("x", []) + score("y", ["x"])
    backward = score("y", []) + score("x", ["y"])
    assert forward - backward > search.margin


@then("x's score equals a node fitted on the 800 rows where x was not set")
def fitted_on_kept_rows(context):
    score = GraphSearch(Background()).score(context["data"], context["intervened"])
    kept = ~context["intervened"]["x"]
    node = GraphSearch(Background()).maker().make("x", ["y"])
    node.fit({name: values[kept] for name, values in context["data"].items()})
    assert np.isclose(score("x", ["y"]), node.evidence)


@then("y's score uses all 1000 rows")
def all_rows(context):
    score = GraphSearch(Background()).score(context["data"], context["intervened"])
    node = GraphSearch(Background()).maker().make("y", ["x"])
    node.fit(context["data"])
    assert np.isclose(score("y", ["x"]), node.evidence)

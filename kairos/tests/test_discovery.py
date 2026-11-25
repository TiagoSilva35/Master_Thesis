"""Steps for discovery.feature."""

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.discovery import Background, Graph, GraphSearch
from src.simulation.world import Equation, Outcome, Uniform, World
from tests.helpers import sigmoid, world

scenarios("discovery.feature")


def names(text: str) -> set[str]:
    """'age, education and debt' -> {'age', 'education', 'debt'}"""
    return {name.strip() for name in text.replace(" and ", ", ").split(",")}


def arrow(text: str) -> tuple[str, str]:
    """'education → Y' -> ('education', 'Y')"""
    parent, child = text.split("→")
    return parent.strip(), child.strip()


# Given

@given(parsers.parse("{count:d} people from world {number:d}"))
def world_people(context, count, number):
    context["world"] = world(number - 1).world
    context["data"] = context["world"].sample(count, seed=0)


@given(parsers.parse("{count:d} people where income → savings → Y, with no arrow income → Y"))
def chain_people(context, count):
    chain = World("chain", {
        "income": Uniform(0.0, 4.0),
        "savings": Equation(0.0, {"income": lambda x: 0.5 * x**2}, noise=0.3),
        "Y": Outcome(-2.0, {"savings": lambda x: 0.5 * x}),
    })
    context["data"] = chain.sample(count, seed=0)


@given(parsers.parse("{count:d} people where a is bell-shaped, b = 2·a plus noise, and b → Y"))
def straight_line_people(context, count):
    random = np.random.default_rng(0)
    a = random.normal(size=count)
    b = 2.0 * a + 0.5 * random.normal(size=count)
    y = (random.uniform(size=count) < sigmoid(b)).astype(float)
    context["data"] = {"a": a, "b": b, "Y": y}


@given(parsers.parse("{count:d} people from a random {size:d}-variable graph with curved arrows and non-bell noise"))
def random_graph_people(context, count, size):
    """Each variable after the first gets each earlier one as a parent with chance
    0.35, through a tanh curve; its noise is evenly spread or heavy-tailed."""
    random = np.random.default_rng(1)
    data, truth = {}, set()
    for child in range(size):
        parents = [parent for parent in range(child) if random.uniform() < 0.35]
        noise = random.uniform(-1, 1, count) if random.uniform() < 0.5 else random.laplace(0, 0.5, count)
        value = noise + sum(np.tanh(random.uniform(0.5, 1.5) * data[f"v{parent}"]) * random.choice([-1, 1])
                            for parent in parents)
        data[f"v{child}"] = (value - value.mean()) / value.std()
        truth |= {(f"v{parent}", f"v{child}") for parent in parents}
    context["data"], context["truth"] = data, truth


@given("the background says nothing")
def no_background(context):
    context["background"] = Background()


@given("the background says only that Y is the outcome")
def outcome_only(context):
    context["background"] = Background("Y")


@given(parsers.re(r"the background says Y is the outcome and (?P<roots>.+) (?:is a root|are roots)"))
def outcome_and_roots(context, roots):
    context["background"] = Background("Y", names(roots))


@given(parsers.parse("a graph with the arrows {first} and {second}"))
def small_graph(context, first, second):
    graph = Graph(["a", "b", "c"])
    graph.add(*arrow(first))
    graph.add(*arrow(second))
    context["graph"] = graph


# When

@when("the search runs")
def search(context):
    context["found"] = GraphSearch(context["background"]).run(context["data"])


@when("the search climbs from a graph where Y has the parents income and savings")
def climb_from_two_parents(context):
    search = GraphSearch(context["background"])
    graph = Graph(sorted(context["data"]))
    graph.add("income", "savings")
    graph.add("income", "Y")
    graph.add("savings", "Y")
    context["graph"] = search.climb(graph, search.score(context["data"]))


@when("the search climbs once, and then with restarts")
def climb_and_restart(context):
    search, variables = GraphSearch(context["background"]), sorted(context["data"])
    score = search.score(context["data"])
    first, best = search.climb(Graph(variables), score), search.best_graph(variables, score)
    context["worths"] = search.worth(first, score), search.worth(best, score)
    context["best"] = best


@when("savings is scored twice with income as its parent")
def score_twice(context):
    score = GraphSearch(Background("Y")).score(context["data"])
    score("savings", ["income"])
    score("savings", ["income"])
    context["score"] = score


@when("kairos is built from the discovered graph and fitted")
def build_kairos(context):
    model = context["found"].build()
    model.fit(context["data"])
    context["model"] = model


# Then

@then("the arrows found are the world's arrows")
def same_arrows(context):
    assert set(context["found"].arrows()) == set(context["world"].edges())


@then(parsers.parse("the arrows found are the world's arrows except {missing}"))
def same_arrows_except(context, missing):
    assert set(context["found"].arrows()) == set(context["world"].edges()) - {arrow(missing)}


@then(parsers.parse("{name} has no parents"))
def no_parents(context, name):
    assert context["found"].parents[name] == []


@then(parsers.parse("{name} is no variable's parent"))
def never_a_parent(context, name):
    assert all(parent != name for parent, _ in context["found"].arrows())


@then(parsers.parse("there is no arrow {text}"))
def no_arrow(context, text):
    assert arrow(text) not in context["found"].arrows()


@then("Y keeps savings and loses income")
def pruned(context):
    assert context["graph"].parents["Y"] == ["savings"]


@then("the restarts' graph is worth at least as much as the first one")
def restarts_never_lose(context):
    first, best = context["worths"]
    assert best >= first


@then("the restarts' arrows are the true ones")
def restarts_find_truth(context):
    assert set(context["best"].arrows()) == context["truth"]


@then("every arrow's gain is above the price")
def gains_above_price(context):
    assert min(context["found"].gains.values()) > GraphSearch(Background()).price


@then("the arrow between a and b is flagged as doubtful")
def flagged(context):
    assert {("a", "b"), ("b", "a")} & set(context["found"].doubtful)


@then(parsers.parse("the arrow {text} is not flagged"))
def not_flagged(context, text):
    assert arrow(text) in context["found"].arrows()
    assert arrow(text) not in context["found"].doubtful


@then(parsers.parse("the arrow {text} would close a loop"))
def closes_loop(context, text):
    assert context["graph"].creates_loop(*arrow(text))


@then(parsers.parse("the arrow {text} would not"))
def no_loop(context, text):
    assert not context["graph"].creates_loop(*arrow(text))


@then(parsers.parse("the order is {text}"))
def order(context, text):
    assert context["graph"].order() == [name.strip() for name in text.split(",")]


@then("one fit was run")
def one_fit(context):
    assert context["score"].fits == 1


@then(parsers.parse("its chances of Y = 1 are within {tolerance:g} of the true chances on average"))
def chances_close(context, tolerance):
    predicted = context["model"].node("Y").predict(context["data"])
    assert np.mean(np.abs(predicted - context["world"].chance(context["data"]))) < tolerance

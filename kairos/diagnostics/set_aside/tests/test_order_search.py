"""Steps for order_search.feature."""

import random

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from src.causal_model.discovery import Background, Graph, GraphSearch, OrderSearch, ParentSets

scenarios("order_search.feature")


def random_graph(seed: int, rows: int, size: int) -> tuple[dict, set]:
    """Each variable after the first gets each earlier one as a parent with chance
    0.35, through a tanh curve; its noise is evenly spread or heavy-tailed."""
    random_ = np.random.default_rng(seed)
    data, truth = {}, set()
    for child in range(size):
        parents = [parent for parent in range(child) if random_.uniform() < 0.35]
        noise = random_.uniform(-1, 1, rows) if random_.uniform() < 0.5 else random_.laplace(0, 0.5, rows)
        value = noise + sum(np.tanh(random_.uniform(0.5, 1.5) * data[f"v{parent}"]) * random_.choice([-1, 1])
                            for parent in parents)
        data[f"v{child}"] = (value - value.mean()) / value.std()
        truth |= {(f"v{parent}", f"v{child}") for parent in parents}
    return data, truth


def total(score, graph: Graph) -> float:
    return sum(score(name, list(graph.parents[name])) for name in graph.variables)


@given(parsers.parse("{count:d} people where z = x + y plus bell-shaped noise, with x and y unrelated (seed {seed:d})"))
def collider(context, count, seed):
    random_ = np.random.default_rng(seed)
    x, y = random_.normal(0, 1, count), random_.normal(0, 1, count)
    context["data"] = {"x": x, "y": y, "z": x + y + random_.normal(0, 0.5, count)}
    context["search"] = GraphSearch(Background())


@given(parsers.parse("{count:d} people from a random {size:d}-variable graph with curved arrows and non-bell noise "
                     "(seed {seed:d})"))
def random_people(context, count, size, seed):
    context["data"], context["truth"] = random_graph(seed, count, size)
    context["search"] = GraphSearch(Background(), noise_floor=0.01, learn_noise=True)


@when("the greedy search runs, and then the order search from its graph")
def run_both(context):
    search, data = context["search"], context["data"]
    score = search.score(data)
    greedy = Graph(sorted(data))
    search.add_arrows(greedy, score)
    context["score"], context["greedy"] = score, greedy
    context["order"] = OrderSearch().improve(greedy, score, Background())


@when("parent sets are prepared with v0 as a root")
def prepare(context):
    background = Background(roots={"v0"})
    context["score"] = context["search"].score(context["data"])
    context["sets"] = ParentSets(context["score"], background)
    context["sets"].prepare(sorted(context["data"]))


@then("greedy does not find x → z ← y")
def greedy_trapped(context):
    assert set(context["greedy"].arrows()) != {("x", "z"), ("y", "z")}


@then("the order search finds x → z ← y, with a higher total evidence")
def order_escapes(context):
    assert set(context["order"].arrows()) == {("x", "z"), ("y", "z")}
    assert total(context["score"], context["order"]) > total(context["score"], context["greedy"])


@then("greedy's arrows differ from the true ones")
def greedy_wrong(context):
    assert set(context["greedy"].arrows()) != context["truth"]


@then("the order search's arrows are the true ones")
def order_right(context):
    assert set(context["order"].arrows()) == context["truth"]


@then("the order search never scores below greedy")
def never_worse(context):
    assert total(context["score"], context["order"]) >= total(context["score"], context["greedy"]) - 1e-9


@then("v0's only option is no parents")
def root_option(context):
    assert [parents for _, parents in context["sets"].options["v0"]] == [()]


@then("each variable's best option within a set uses only parents from that set and beats every small set within it")
def best_within(context):
    sets, coin, score = context["sets"], random.Random(0), context["score"]
    names = sorted(context["data"])
    for name in names:
        allowed = {other for other in names if other != name and coin.random() < 0.5}
        evidence, parents = sets.best(name, allowed)
        assert set(parents) <= allowed
        assert evidence == score(name, list(parents))
        assert evidence >= max(value for value, option in sets.options[name] if set(option) <= allowed)


@then("swapping each pair of neighbours in an order changes the total by the swap's computed change")
def swap_change(context):
    search, sets = OrderSearch(), context["sets"]
    order = sorted(context["data"])
    for i in range(len(order) - 1):
        swapped = list(order)
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        assert np.isclose(search.total(swapped, sets) - search.total(order, sets), search._swap_change(order, i, sets))

"""Steps for discovery_baselines.feature."""

from fractions import Fraction

import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from pytest_bdd import given, parsers, scenarios, then, when

from src.benchmarks.causal_arena import FoundGraph, GraphScore
from src.benchmarks.causal_arena.methods import CausalLearnGraph, CDNODSearch

scenarios("discovery_baselines.feature")


def arrows(text: str) -> set[tuple[str, str]]:
    """'a → b, b → c and c → d' -> {('a', 'b'), ('b', 'c'), ('c', 'd')}"""
    found = set()
    for part in text.replace(" and ", ", ").split(","):
        parent, child = part.split("→")
        found.add((parent.strip(), child.strip()))
    return found


def pairs(text: str) -> set[tuple[str, str]]:
    """'c — d and a — e' -> {('c', 'd'), ('a', 'e')}, each pair with its names sorted"""
    found = set()
    for part in text.replace(" and ", ", ").split(","):
        first, second = (name.strip() for name in part.split("—"))
        found.add((min(first, second), max(first, second)))
    return found


# --- scoring -------------------------------------------------------------------------

@given(parsers.parse("the true arrows {text}"))
def true_arrows(context, text):
    context["truth"] = arrows(text)


@given(parsers.parse("the found arrows {text}"))
def found_arrows(context, text):
    context["score"] = GraphScore.compare(arrows(text), context["truth"])


@then(parsers.parse("{right:d} arrow is right, {turned:d} is turned around, {extra:d} is extra and {missed:d} is missed"))
def counts(context, right, turned, extra, missed):
    score = context["score"]
    assert (score.correct, score.reversed, score.extra, score.missing) == (right, turned, extra, missed)


@then(parsers.parse("the precision is {precision} and the recall is {recall}"))
def precision_recall(context, precision, recall):
    assert Fraction(context["score"].precision).limit_denominator(100) == Fraction(precision)
    assert Fraction(context["score"].recall).limit_denominator(100) == Fraction(recall)


@then(parsers.parse("the structural Hamming distance is {distance:d}"))
def hamming(context, distance):
    assert context["score"].shd == distance


@given(parsers.parse("a graph with the arrow {arrow} and the undirected pairs {undirected}"))
def partly_undirected(context, arrow, undirected):
    context["found"] = FoundGraph(arrows(arrow), pairs(undirected), rows=0, seconds=0.0)


@then(parsers.parse("at best {right:d} arrows are right, {turned:d} are turned around and {extra:d} is extra"))
def best_case(context, right, turned, extra):
    assert counts_when_oriented(context, "best") == (right, turned, extra)


@then(parsers.parse("at worst {right:d} arrow is right, {turned:d} is turned around and {extra:d} is extra"))
def worst_case(context, right, turned, extra):
    assert counts_when_oriented(context, "worst") == (right, turned, extra)


def counts_when_oriented(context, how: str) -> tuple[int, int, int]:
    score = GraphScore.compare(context["found"].oriented(context["truth"], how), context["truth"])
    return score.correct, score.reversed, score.extra


# --- causal-learn's graphs, PC and CD-NOD -------------------------------------------------

@given("a causal-learn matrix over x, y, z and w plus a changing factor, with x → y, z — w and y → factor")
def causal_learn_matrix(context):
    matrix = np.zeros((5, 5))
    matrix[1, 0], matrix[0, 1] = 1, -1    # x → y
    matrix[2, 3], matrix[3, 2] = -1, -1   # z — w
    matrix[4, 1], matrix[1, 4] = 1, -1    # y → factor, which is not a variable
    context["read"] = CausalLearnGraph.read(matrix, ["x", "y", "z", "w"])


@then(parsers.parse("the arrows read are {text}"))
def arrows_read(context, text):
    assert context["read"][0] == arrows(text)


@then(parsers.parse("the undirected pairs read are {text}"))
def pairs_read(context, text):
    assert context["read"][1] == pairs(text)


@given(parsers.parse("{count:d} rows where c = a + b plus noise and a and b are independent"))
def collider_rows(context, count):
    random = np.random.default_rng(0)
    a, b = random.normal(size=count), random.normal(size=count)
    context["names"] = ["a", "b", "c"]
    context["matrix"] = np.column_stack([a, b, a + b + 0.5 * random.normal(size=count)])


@given(parsers.parse("two runs of {count:d} rows where x shifts between the runs and y = 2·x plus noise"))
def shifted_runs(context, count):
    random = np.random.default_rng(0)
    run = np.repeat([0.0, 1.0], count)
    x = 1.5 * run + random.normal(size=2 * count)
    context["names"] = ["x", "y"]
    context["matrix"] = np.column_stack([x, 2 * x + random.normal(size=2 * count)])
    context["factor"] = run[:, None]


@when("PC with the Fisher z test runs without background")
def run_pc(context):
    graph = pc(context["matrix"], 0.05, "fisherz", show_progress=False)
    context["read"] = CausalLearnGraph.read(graph.G.graph, context["names"])


@when("CD-NOD with the Fisher z test runs on the two runs")
def run_cdnod(context):
    # α = 0.01: at 0.05 the test that y is independent of the run given x fails by
    # chance for this sample (p = 0.042), which keeps a run → y link and blocks the orientation
    graph = CDNODSearch.search(context["matrix"], context["factor"], 0.01, "fisherz", None)
    context["read"] = CausalLearnGraph.read(graph.G.graph, context["names"])


@when("CD-NOD with the Fisher z test runs on the two runs with x and y as roots")
def run_cdnod_with_roots(context):
    knowledge = CausalLearnGraph.no_arrow_into({"x", "y"}, context["names"])
    graph = CDNODSearch.search(context["matrix"], context["factor"], 0.01, "fisherz", knowledge)
    context["read"] = CausalLearnGraph.read(graph.G.graph, context["names"])


@then(parsers.parse("the arrows found are {text}"))
def arrows_found(context, text):
    assert context["read"][0] == arrows(text)


@then("x and y are not linked")
def not_linked(context):
    directed, undirected = context["read"]
    assert not directed and not undirected

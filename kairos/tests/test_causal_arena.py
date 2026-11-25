"""Steps for causal_arena.feature. A fake repository serves small files with the
same layout as CausalArena's release (manifest, graph.json, .npz samples)."""

import io
import json
from pathlib import Path

import numpy as np
from pytest_bdd import given, parsers, scenarios, then, when

from src.benchmarks.causal_arena import ArenaBenchmark, ArenaData, ArenaFiles, FoundGraph, GraphScore, ScenarioIndex
from src.benchmarks.causal_arena.methods import KairosInputs
from src.benchmarks.causal_arena.parallel import ParallelRun
from src.benchmarks.causal_arena.scenarios import MANIFEST
from tests.helpers import EmptyMethod

scenarios("causal_arena.feature")

DOMAIN, SLUG, FOLDER = "finance_and_credit", "toy_loans", "001_toy_loans"
NAMES = ["a", "b", "c"]
MEANS, SPREADS = np.array([10.0, 0.5, 2.0]), np.array([2.0, 0.5, 1.0])


def data_path(regime: str) -> str:
    return f"scm/semantic/data/semantic_{regime}/semantic_{DOMAIN}_{SLUG}__semantic_{regime}__rep000.npz"


def sample(regime: str) -> bytes:
    """Six rows of a → b → c; in obs_int the last two rows set b by intervention."""
    raw = np.array([[8, 0, 1], [9, 1, 2], [10, 0, 2], [11, 1, 3], [12, 1, 4], [13, 1, 4]], dtype=float)
    flags = np.zeros_like(raw)
    if regime == "obs_int":
        flags[4:, 1] = 1
    graph = np.zeros((3, 3))
    graph[0, 1] = graph[1, 2] = 1  # row = cause
    metadata = {"standardization": {name: {"mean": m, "std": s} for name, m, s in zip(NAMES, MEANS, SPREADS)}}
    buffer = io.BytesIO()
    np.savez(buffer, x=np.stack([(raw - MEANS) / SPREADS, flags], axis=-1).astype(np.float32), g=graph,
             mask=np.ones_like(raw), node_ids=np.array(NAMES), metadata_json=np.array(json.dumps(metadata)))
    return buffer.getvalue()


def repository() -> dict[str, bytes]:
    manifest = {"domains": {DOMAIN: [{"scenario_dir": FOLDER, "data_slug": SLUG, "data_files": [
        path.removeprefix("scm/semantic/") for path in (data_path("obs_only"), data_path("obs_int"))]}]}}
    graph = {"nodes": [{"id": "a", "measurement_scale": "continuous"}, {"id": "b", "measurement_scale": "binary"},
                       {"id": "c", "measurement_scale": "ordinal_categorical"}]}
    return {MANIFEST: json.dumps(manifest).encode(),
            f"scm/semantic/artifacts/{DOMAIN}/{FOLDER}/graph.json": json.dumps(graph).encode(),
            data_path("obs_only"): sample("obs_only"), data_path("obs_int"): sample("obs_int")}


@given("a fake repository with one scenario")
def fake_repository(context, tmp_path):
    served = repository()
    context["fetched"] = []

    def fetch(relative: str, local: Path) -> None:
        context["fetched"].append(relative)
        local.write_bytes(served[relative])

    context["files"] = ArenaFiles(tmp_path, fetch)
    context["scenario"] = ScenarioIndex.load(context["files"])[0]


# --- files and data --------------------------------------------------------------

@when("the release manifest is asked for twice")
def manifest_twice(context):
    context["files"].path(MANIFEST)
    context["files"].path(MANIFEST)


@then("it was fetched once")
def fetched_once(context):
    assert context["fetched"].count(MANIFEST) == 1


@when(parsers.parse("its {kind} file is loaded"))
def load_file(context, kind):
    regime = "obs_int" if kind == "interventional" else "obs_only"
    context["data"] = ArenaData(context["files"]).load(context["scenario"], regime, 0)


@then("the true arrows are a → b and b → c")
def true_arrows(context):
    assert context["data"].truth == {("a", "b"), ("b", "c")}


@then("the raw values are recovered from the standardised ones")
def raw_values(context):
    assert np.allclose(context["data"].raw[:, 0], [8, 9, 10, 11, 12, 13], atol=1e-4)
    assert np.allclose(context["data"].raw[:, 1], [0, 1, 0, 1, 1, 1], atol=1e-4)


@then("the measurement scales are continuous for a, binary for b and ordinal_categorical for c")
def scales(context):
    assert context["data"].kinds == {"a": "continuous", "b": "binary", "c": "ordinal_categorical"}


@then("C is 0 for the 4 observational rows and 1 for the 2 rows that set b")
def changing_factor(context):
    assert context["data"].change.tolist() == [0, 0, 0, 0, 1, 1]


@then("the variables without a cause are a")
def roots(context):
    assert context["data"].roots == {"a"}


@then("there is no C")
def no_change(context):
    assert context["data"].change is None and context["data"].change_kind == "none"


# --- what kairos gets ---------------------------------------------------------------

@then("kairos treats a as continuous, b as binary and c as ordinal")
def kairos_kinds(context):
    assert KairosInputs(context["data"]).kinds == {"a": "continuous", "b": "binary", "c": "ordinal"}


@then("kairos gets the codes 0, 1, 0, 1, 1, 1 for b and 0, 1, 1, 2, 3, 3 for c")
def kairos_codes(context):
    table = KairosInputs(context["data"]).table()
    assert table["b"].tolist() == [0, 1, 0, 1, 1, 1]
    assert table["c"].tolist() == [0, 1, 1, 2, 3, 3]
    assert table["a"].tolist() == [8, 9, 10, 11, 12, 13]  # continuous values in their own units


@when(parsers.parse('the scale "{scale}" with {levels:d} distinct values is read'))
def read_scale(context, scale, levels):
    context["kind"] = KairosInputs.kind(scale, levels)


@then(parsers.parse("kairos treats that variable as {kind}"))
def treated_as(context, kind):
    assert context["kind"] == kind


# --- orientation of undirected pairs ---------------------------------------------

@given("the true arrows a → b and c → d and a graph with the undirected pairs a — b, c — d and a — e")
def partly_undirected(context):
    context["truth"] = {("a", "b"), ("c", "d")}
    context["found"] = FoundGraph(set(), {("a", "b"), ("c", "d"), ("a", "e")}, rows=0, seconds=0.0)


@then("pointing the pairs twice with the same seed gives the same arrows")
def same_seed(context):
    found = context["found"]
    assert found.oriented(context["truth"], "random", 7) == found.oriented(context["truth"], "random", 7)


@then("the precision with random directions lies between the worst and the best")
def between(context):
    found, truth = context["found"], context["truth"]
    precision = {how: GraphScore.compare(found.oriented(truth, how, 7), truth).precision
                 for how in ("worst", "random", "best")}
    assert precision["worst"] <= precision["random"] <= precision["best"]


# --- the benchmark loop ------------------------------------------------------------

class StubMethod:
    """A method that returns an empty graph, or raises."""

    def __init__(self, name: str, uses_background: bool, needs_change: bool, fails: bool = False):
        self.name, self.uses_background, self.needs_change, self.fails = name, uses_background, needs_change, fails

    def find(self, data, roots):
        if self.fails:
            raise RuntimeError("cannot fit")
        return FoundGraph(set(), set(), len(data.values), 0.0)


@when("a method that takes background, one that does not, and one that needs interventions run on both files")
def run_loop(context):
    methods = [StubMethod("with background", True, False), StubMethod("without", False, False),
               StubMethod("needs C", True, True)]
    benchmark = ArenaBenchmark(ArenaData(context["files"]), methods)  # type: ignore[arg-type]
    context["runs"] = {regime: list(benchmark.run(context["scenario"], regime, 0)) for regime in ("obs_only", "obs_int")}


@then("the observational file has 3 runs and the interventional file has 5")
def run_counts(context):
    # obs_only: 2 backgrounds + 1; obs_int: the same plus 2 for the method that needs C
    assert len(context["runs"]["obs_only"]) == 3 and len(context["runs"]["obs_int"]) == 5


@given("a fake repository with one scenario, already downloaded")
def downloaded_repository(context, tmp_path):
    for path, content in repository().items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(content)
    files = ArenaFiles(tmp_path)
    scenario = ScenarioIndex.load(files)[0]
    context["benchmark"] = ArenaBenchmark(ArenaData(files), [EmptyMethod()])
    context["tasks"] = [(scenario, regime, 0) for regime in ("obs_only", "obs_int")]
    context["folder"] = tmp_path / "kept"


def run_files(context, tasks, folder, workers=1) -> list[list]:
    return list(ParallelRun(context["benchmark"], context["folder"] / folder, workers).results(tasks))


@when(parsers.parse("both files run with 1 worker and with {workers:d} workers"))
def run_both_ways(context, workers):
    context["one"] = run_files(context, context["tasks"], "one")
    context["many"] = run_files(context, context["tasks"], "many", workers)


@then("the two runs have the same results")
def same_results(context):
    def rows(runs):
        return sorted((r.key, r.method, r.background, r.score.f1) for found in runs for r in found)
    assert len(context["many"]) == 2 and rows(context["one"]) == rows(context["many"])


@when("the observational file runs, and then both files with the same folder")
def run_again(context):
    run_files(context, context["tasks"][:1], "kept")
    asked = []
    original = context["benchmark"].run
    context["benchmark"].run = lambda *task: asked.append(task) or original(*task)
    context["second"], context["asked"] = run_files(context, context["tasks"], "kept"), asked


@then("the second run reads the observational file's results back and runs only the other file")
def resumed(context):
    assert len(context["second"]) == 2 and context["asked"] == context["tasks"][1:]


@when("a failing method and a working one run on the observational file")
def run_failing(context):
    methods = [StubMethod("broken", True, False, fails=True), StubMethod("fine", True, False)]
    benchmark = ArenaBenchmark(ArenaData(context["files"]), methods)  # type: ignore[arg-type]
    context["results"] = list(benchmark.run(context["scenario"], "obs_only", 0))


@then("the failing method's runs keep its error and the working method's runs have a graph")
def failure_recorded(context):
    broken = [r for r in context["results"] if r.method == "broken"]
    fine = [r for r in context["results"] if r.method == "fine"]
    assert all(r.found is None and "cannot fit" in r.error for r in broken)
    assert all(r.found is not None for r in fine) and len(fine) == 2

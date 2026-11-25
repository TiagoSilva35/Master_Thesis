"""Smoke test: kairos on rep 0 of two CausalArena scenarios, both regimes, both backgrounds.

Run from anywhere:  python -I smoke.py <data folder> [--describe]
"""

import sys
import time
from pathlib import Path

KAIROS = Path("/Users/tiagosilva/Documents/Master_Thesis/kairos")
sys.path.insert(0, str(KAIROS))

import numpy as np  # noqa: E402

from src.benchmarks.causal_arena import ArenaBenchmark, ArenaData, ArenaFiles, ScenarioIndex  # noqa: E402
from src.benchmarks.causal_arena import GraphScore  # noqa: E402
from src.benchmarks.causal_arena.methods import KairosInputs, KairosSearch  # noqa: E402

PICKS = (("finance_and_credit", "mortgage"), ("healthcare_delivery", ""))


def chosen(scenarios):
    picked = []
    for domain, word in PICKS:
        picked.append(next(s for s in scenarios if s.domain == domain and word in s.slug))
    return picked


def describe(data) -> None:
    for column, name in enumerate(data.variables):
        raw = data.raw[:, column]
        levels = np.unique(np.round(raw, 6))
        integer = np.allclose(levels, np.round(levels))
        low, high = raw.min(), raw.max()
        piled = max(np.mean(np.isclose(raw, low)), np.mean(np.isclose(raw, high)))
        print(f"  {name:40s} {data.kinds[name]:22s} levels {len(levels):5d} "
              f"{'integers ' + str(levels[:6].astype(int).tolist()) if integer and len(levels) <= 12 else ''}"
              f"{'  piled at a bound: %.0f%%' % (100 * piled) if piled > 0.05 else ''}")


def main() -> None:
    if "--no-interventions" in sys.argv:
        KairosInputs.intervened = lambda self: None  # Phase 4 alone: obs_int rows fitted as if observed
    files = ArenaFiles(Path(sys.argv[1]))
    scenarios = chosen(ScenarioIndex.load(files))
    method = KairosSearch(search="order") if "--order" in sys.argv else KairosSearch(noise_floor=float(sys.argv[2]), learn_noise=sys.argv[3] == "learnt")
    benchmark = ArenaBenchmark(ArenaData(files), [method])
    for scenario in scenarios:
        for regime in ("obs_only", "obs_int"):
            if "--describe" in sys.argv:
                data = ArenaData(files).load(scenario, regime, 0)
                print(f"{data.key}: {len(data.variables)} variables")
                describe(data)
                continue
            for result in benchmark.run(scenario, regime, 0):
                if result.found is None:
                    print(f"{result.key} · {result.background}: failed ({result.error})")
                    continue
                score = result.score
                data = ArenaData(files).load(scenario, regime, 0)
                kinds = KairosInputs(data).kinds
                found = result.graph.directed
                into = GraphScore.compare({a for a in found if kinds[a[1]] != "continuous"},
                                          {a for a in result.truth if kinds[a[1]] != "continuous"})
                out = GraphScore.compare({a for a in found if kinds[a[0]] != "continuous"},
                                         {a for a in result.truth if kinds[a[0]] != "continuous"})
                spread = sorted(a for a in found if "current_rate_spread" in a)
                doubtful_spread = [a for a in result.graph.doubtful if "current_rate_spread" in a]
                print(f"{result.key} · {result.background}: F1 {score.f1:.2f} (precision {score.precision:.2f}, "
                      f"recall {score.recall:.2f}, correct {score.correct}, reversed {score.reversed}, "
                      f"extra {score.extra}, missing {score.missing}), doubtful {len(result.graph.doubtful)}, "
                      f"{result.graph.seconds:.0f}s", flush=True)
                print(f"    into categorical: F1 {into.f1:.2f} ({into.correct}/{into.true} correct, {into.reversed} reversed); "
                      f"out of categorical: F1 {out.f1:.2f} ({out.correct}/{out.true} correct, {out.reversed} reversed)")
                if spread:
                    print(f"    spread arrows: {spread}; doubtful: {doubtful_spread}")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"total {time.perf_counter() - start:.0f}s")

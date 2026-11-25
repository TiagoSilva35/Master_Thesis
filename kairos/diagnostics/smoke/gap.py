"""How far the graph kairos finds is ahead of the true one, in worth (0 or less: the score points to the truth, or the
search stopped below it), and where climbing from the truth ends up. Observational file, no background.

From kairos/:  venv/bin/python -I diagnostics/smoke/gap.py <mortgage or medication>
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
KAIROS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KAIROS))

from src.benchmarks.causal_arena import ArenaData, ArenaFiles, GraphScore, ScenarioIndex  # noqa: E402
from src.benchmarks.causal_arena.methods import KairosInputs  # noqa: E402
from src.causal_model.discovery import Background, Graph, GraphSearch  # noqa: E402

files = ArenaFiles(KAIROS / "data" / "causal_arena")
scenario = next(s for s in ScenarioIndex.load(files) if sys.argv[1] in s.slug)
data = ArenaData(files).load(scenario, "obs_only", 0)
inputs = KairosInputs(data)
table = inputs.table()
search = GraphSearch(Background(kinds=inputs.kinds))
score = search.score(table)
truth = Graph(sorted(table))
for parent, child in sorted(data.truth):
    truth.add(parent, child)
found, near = search.best_graph(sorted(table), score), search.climb(truth, score)
kept = search.gains(truth, score)
print(f"{sys.argv[1]}: worth found − truth {search.worth(found, score) - search.worth(truth, score):.0f}; "
      f"climbing from the truth gains {search.worth(near, score) - search.worth(truth, score):.0f}")
print(f"F1 found {GraphScore.compare(set(found.arrows()), data.truth).f1:.2f}, climbing from the truth "
      f"{GraphScore.compare(set(near.arrows()), data.truth).f1:.2f}; true arrows that earn their price "
      f"{sum(gain > search.price for gain in kept.values())} of {len(kept)}")
for gap, name in sorted(((search._value(score, v, found.parents[v]) - search._value(score, v, truth.parents[v]), v)
                         for v in truth.variables), key=lambda row: -abs(row[0]))[:8]:
    print(f"   {gap:8.0f} {name:32s} found {[p[:18] for p in found.parents[name]]} truth {[p[:18] for p in truth.parents[name]]}")

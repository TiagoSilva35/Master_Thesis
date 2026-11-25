"""Pairwise direction on true edges: for each true arrow a → b, margin =
[s(a|∅) + s(b|a)] − [s(b|∅) + s(a|b)] under several scorings. Also each variable's
'shape bonus' as a root: learnt-noise root score − bell root score + N/2(log 2π + 1)."""
import sys
sys.path.insert(0, "/Users/tiagosilva/Documents/Master_Thesis/kairos")
from pathlib import Path
import numpy as np
from src.benchmarks.causal_arena import ArenaData, ArenaFiles, ScenarioIndex
from src.benchmarks.causal_arena.methods import KairosInputs
from src.causal_model.discovery import EvidenceScore, NodeMaker
from src.causal_model.nodes import BayesianFitting

files = ArenaFiles(Path(sys.argv[1]))
word = sys.argv[2]
s = next(x for x in ScenarioIndex.load(files) if word in x.slug)
d = ArenaData(files).load(s, "obs_only", 0)
inputs = KairosInputs(d)
table, kinds = inputs.table(), inputs.kinds
fit = BayesianFitting()
configs = {
    "P1 all bell": EvidenceScore(table, NodeMaker({}, None, fit, 8, noise_floor=0.01)),
    "P3 kinds, bell": EvidenceScore(table, NodeMaker(kinds, None, fit, 8, noise_floor=0.01)),
    "P4 kinds, learnt": EvidenceScore(table, NodeMaker(kinds, None, fit, 8, noise_floor=0.01, learn_noise=True)),
    "P4 no floor": EvidenceScore(table, NodeMaker(kinds, None, fit, 8, noise_floor=0.0, learn_noise=True)),
}
def margin(score, a, b):
    return score(a, []) + score(b, [a]) - score(b, []) - score(a, [b])

rows = []
for a, b in sorted(d.truth):
    rows.append((a, b, {name: margin(score, a, b) for name, score in configs.items()}))
def group(label, keep):
    sel = [r for r in rows if keep(r[0], r[1])]
    if not sel: return
    line = f"{label:34s} {len(sel):3d} edges |"
    for name in configs:
        m = np.array([r[2][name] for r in sel])
        line += f" {name}: {np.mean(m > 0):.2f} right ({np.sum(m > 3)}✓ {np.sum(m < -3)}✗) |"
    print(line)
cont = lambda v: kinds[v] == "continuous"
print(word, {k: sum(v == k for v in kinds.values()) for k in ("continuous", "binary", "ordinal", "nominal")})
group("all true edges", lambda a, b: True)
group("continuous → continuous", lambda a, b: cont(a) and cont(b))
group("continuous → categorical", lambda a, b: cont(a) and not cont(b))
group("categorical → continuous", lambda a, b: not cont(a) and cont(b))
group("categorical → categorical", lambda a, b: not cont(a) and not cont(b))
print()
print("edge margins (P1 bell, P3 kinds bell, P4 learnt):")
for a, b, m in rows:
    print(f"  {a[:28]:28s} ({kinds[a][:4]}) → {b[:28]:28s} ({kinds[b][:4]})  "
          + "  ".join(f"{m[n]:8.1f}" for n in ("P1 all bell", "P3 kinds, bell", "P4 kinds, learnt")))
print()
n = len(table[d.variables[0]])
learnt, bell = configs["P4 kinds, learnt"], configs["P3 kinds, bell"]
print("root shape bonus (learnt − bell + N/2(log2π+1)), continuous variables:")
for v in d.variables:
    if cont(v):
        print(f"  {v:40s} {learnt(v, []) - bell(v, []) + n / 2 * (np.log(2 * np.pi) + 1):8.1f}")

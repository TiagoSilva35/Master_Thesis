from itertools import combinations

from src.causal_model.discovery.background import Background
from src.causal_model.discovery.evidence_score import EvidenceScore

Choice = tuple[float, tuple[str, ...]]  # (evidence, parents)


class ParentSets:
    """For each variable, the best parent set among any allowed variables (those
    before it in an order), for a search over orders.

    1. Candidates: the `candidates` variables whose arrow into the variable gains
       most on its own (and gains at all), among those the background allows. This
       is the preliminary neighbourhood selection of CAM (Bühlmann et al. 2014).
    2. Small sets, scored once: every set of at most `most` candidates, plus any
       extra sets handed in (the greedy search's own parents, so a search that
       starts from greedy's order can never do worse than greedy).
    3. Grown sets: among the allowed candidates, add the parent that raises the
       evidence most while one does, then drop any parent whose removal raises it
       (as CAM does within an order). No cap on how many parents. Remembered for
       each set of allowed candidates.
    4. `best(variable, allowed)`: the better of the best small set within
       `allowed` and the grown set.

    Every fit goes through the EvidenceScore, which remembers it, so the same parent
    set is never fitted twice. A root (background) has a single option: no parents.
    """

    def __init__(self, score: EvidenceScore, background: Background, candidates: int = 10, most: int = 2):
        self.score = score
        self.background = background
        self.candidates = candidates
        self.most = most
        self.options: dict[str, list[Choice]] = {}      # small sets, best first
        self.chosen: dict[str, frozenset[str]] = {}     # each variable's candidates
        self._grown: dict[tuple[str, frozenset[str]], Choice] = {}

    def prepare(self, variables: list[str], extra: dict[str, list[str]] | None = None) -> None:
        extra = extra if extra is not None else {}
        for variable in variables:
            if variable in self.background.roots:
                self.chosen[variable] = frozenset()
                self.options[variable] = [(self.score(variable, []), ())]
                continue
            self.chosen[variable] = self._candidates(variable, variables)
            sets = {parents for size in range(self.most + 1)
                    for parents in combinations(sorted(self.chosen[variable]), size)}
            if variable in extra:
                sets.add(tuple(sorted(extra[variable])))
            options = [(self.score(variable, list(parents)), parents) for parents in sets]
            self.options[variable] = sorted(options, key=lambda option: option[0], reverse=True)

    def best(self, variable: str, allowed: set[str]) -> Choice:
        small = next(option for option in self.options[variable] if set(option[1]) <= allowed)  # () always fits
        grown = self._grow(variable, self.chosen[variable] & allowed)
        return grown if grown[0] > small[0] else small

    def _grow(self, variable: str, allowed: frozenset[str]) -> Choice:
        key = (variable, allowed)
        if key not in self._grown:
            parents: list[str] = []
            current = self.score(variable, [])
            while True:
                gains = {other: self.score(variable, parents + [other]) - current for other in allowed - set(parents)}
                best = max(gains, key=lambda other: gains[other], default=None)
                if best is None or gains[best] <= 0:
                    break
                parents.append(best)
                current += gains[best]
            dropped = True
            while dropped and parents:
                losses = {parent: current - self.score(variable, [p for p in parents if p != parent]) for parent in parents}
                weakest = min(losses, key=lambda parent: losses[parent])
                dropped = losses[weakest] < 0
                if dropped:
                    parents.remove(weakest)
                    current -= losses[weakest]
            self._grown[key] = (current, tuple(sorted(parents)))
        return self._grown[key]

    def _candidates(self, variable: str, variables: list[str]) -> frozenset[str]:
        alone = self.score(variable, [])
        gains = {other: self.score(variable, [other]) - alone for other in variables
                 if self.background.allows(other, variable)}
        ranked = sorted((other for other in gains if gains[other] > 0), key=lambda other: gains[other], reverse=True)
        return frozenset(ranked[:self.candidates])

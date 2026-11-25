from dataclasses import dataclass

from src.causal_model.discovery import Arrow


@dataclass(frozen=True)
class GraphScore:
    """A found graph against the true one, arrow by arrow: CausalArena's directed-edge
    precision, recall, F1 and SHD."""

    correct: int    # found, pointing the right way
    reversed: int   # found, pointing the wrong way
    extra: int      # found where the truth has no arrow either way
    missing: int    # true arrows not found in either direction
    found: int
    true: int

    @classmethod
    def compare(cls, found: set[Arrow], truth: set[Arrow]) -> "GraphScore":
        reversed_ = {(a, b) for a, b in found - truth if (b, a) in truth}
        extra = found - truth - reversed_
        missing = {(a, b) for a, b in truth - found if (b, a) not in found}
        return cls(len(found & truth), len(reversed_), len(extra), len(missing), len(found), len(truth))

    @property
    def precision(self) -> float:
        """The share of found arrows that are true."""
        return self.correct / self.found if self.found else 0.0

    @property
    def recall(self) -> float:
        """The share of true arrows that were found."""
        return self.correct / self.true if self.true else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def shd(self) -> int:
        """Structural Hamming distance: arrows to add, remove or turn around."""
        return self.reversed + self.extra + self.missing

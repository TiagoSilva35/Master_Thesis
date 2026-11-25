import numpy as np

from src.benchmarks.causal_arena.data import ScenarioData


class KairosInputs:
    """What kairos needs from one CausalArena file: each variable's kind, its
    values, and the rows where an intervention set it.

    CausalArena describes each variable's measurement scale in words, sometimes
    with a note ("continuous (bounded)", "count (non-negative integer)"), so the
    kind is read from key words:

        binary ...                            → binary
        ... nominal / categorical (unordered) → nominal
        ... ordinal / count / discrete        → ordinal with up to `most_levels`
                                                levels, otherwise continuous
        anything else                         → continuous

    The values are the raw ones for continuous variables (kairos may write an
    equation on a log or log-odds scale, which needs the original zero) and codes
    0, 1, 2, ... for the others: the distinct raw values in increasing order,
    so a level that never occurs gets no code.
    """

    def __init__(self, data: ScenarioData, most_levels: int = 10):
        self.data = data
        self.most_levels = most_levels
        self.kinds = {name: self.kind(data.kinds[name], self._levels(column), most_levels)
                      for column, name in enumerate(data.variables)}

    @staticmethod
    def kind(scale: str, levels: int, most_levels: int = 10) -> str:
        words = scale.lower()
        if words.startswith("binary"):
            return "binary"
        if "nominal" in words or ("categorical" in words and "ordinal" not in words):
            return "nominal"
        if any(word in words for word in ("ordinal", "count", "discrete")):
            return "ordinal" if levels <= most_levels else "continuous"
        return "continuous"

    def table(self) -> dict[str, np.ndarray]:
        """One array per variable: raw values or codes."""
        table = {}
        for column, name in enumerate(self.data.variables):
            if self.kinds[name] == "continuous":
                table[name] = self._values(column)
            else:
                table[name] = self._codes(column)
        return table

    def intervened(self) -> dict[str, np.ndarray] | None:
        """For each variable, True on the rows where an intervention set it;
        None when no row was set (an observational file)."""
        if not self.data.intervened.any():
            return None
        return {name: self.data.intervened[:, column] for column, name in enumerate(self.data.variables)}

    def _levels(self, column: int) -> int:
        return len(np.unique(np.round(self.data.raw[:, column], 6)))

    def _values(self, column: int) -> np.ndarray:
        """The raw values, rounded to 1/100,000 of their spread: the files keep seven
        digits of the standardised value, so a 0 comes back as 0.0001."""
        raw = self.data.raw[:, column]
        spread = raw.std()
        return raw if spread == 0 else np.round(raw, 5 - int(np.floor(np.log10(spread))))

    def _codes(self, column: int) -> np.ndarray:
        _, codes = np.unique(np.round(self.data.raw[:, column], 6), return_inverse=True)
        return codes.astype(float)

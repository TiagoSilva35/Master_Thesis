import numpy as np


class ChiSquareMixture:
    """The null distribution of a KCI statistic: a weighted sum of independent
    χ²(1) draws, simulated `draws` times (the end of UInd_KCItest.m and
    CInd_test_new_withGP_t.m). The p-value is the share of simulated values above
    the statistic.

    When weights × rows reaches a million, MATLAB adds the draws in chunks to save
    memory. The unconditional tests add the last, partial chunk; the conditional
    tests leave it out, which drops the smallest weights. Both are kept as written."""

    def __init__(self, random: np.random.Generator):
        self.random = random

    def p_value(self, weights: np.ndarray, statistic: float, rows: int, draws: int,
                keep_remainder: bool) -> float:
        if len(weights) * rows < 1e6:
            null = weights @ self.random.chisquare(1, (len(weights), draws))
        else:
            null = np.zeros(draws)
            length = max(int(1_000_000 // rows), 100)
            chunks = len(weights) // length
            for chunk in range(chunks):
                part = weights[chunk * length:(chunk + 1) * length]
                null += part @ self.random.chisquare(1, (length, draws))
            rest = weights[chunks * length:]
            if keep_remainder and len(rest):
                null += rest @ self.random.chisquare(1, (len(rest), draws))
        return float(np.mean(null > statistic))

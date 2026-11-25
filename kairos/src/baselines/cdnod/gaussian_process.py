"""gpr_multi.m with covfunc {'covSum', {'covSEard', 'covNoise'}}: Gaussian-process
regression of several outputs on the same inputs. KCI and the direction step use
it only to learn kernel widths, by maximising the marginal likelihood.

GPML code: Copyright (C) 2005 - 2006 by Carl Edward Rasmussen; free to copy, use
and modify for research or education, keeping this notice."""

import numpy as np

from src.baselines.cdnod.minimize import ConjugateGradients
from src.baselines.cdnod.numerics import Numerics


class GaussianProcessWidths:
    """Hyperparameters, all as logs: one length scale per input column, then the
    signal's standard deviation, then the noise's standard deviation."""

    def __init__(self, minimizer: ConjugateGradients):
        self.minimizer = minimizer

    def fit(self, start: np.ndarray, inputs: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        """minimize(start, 'gpr_multi', -350, covfunc, inputs, outputs)."""
        return self.minimizer.minimize(lambda h: self.negative_log_likelihood(h, inputs, outputs),
                                       np.asarray(start, dtype=float))

    @staticmethod
    def negative_log_likelihood(log_hyper: np.ndarray, inputs: np.ndarray,
                                outputs: np.ndarray) -> tuple[float, np.ndarray]:
        """The value and gradient gpr_multi.m returns for m outputs:
        ½ tr(Y'K⁻¹Y) + m Σ log diag(L) + ½ m n log 2π, with K = LL'."""
        rows, columns = inputs.shape
        outputs_count = outputs.shape[1]
        lengths = np.exp(log_hyper[:columns])
        noise_variance = np.exp(2 * log_hyper[columns + 1])
        k_signal = Numerics.se_kernel(inputs, lengths, float(np.exp(log_hyper[columns])))
        lower = np.linalg.cholesky(k_signal + noise_variance * np.eye(rows))  # fails if not positive definite
        inverse_lower = np.linalg.inv(lower)
        inverse = inverse_lower.T @ inverse_lower
        alpha = inverse @ outputs
        value = (0.5 * np.sum(outputs * alpha) + outputs_count * np.sum(np.log(np.diag(lower)))
                 + 0.5 * outputs_count * rows * np.log(2 * np.pi))
        w = outputs_count * inverse - alpha @ alpha.T
        gradient = np.empty(columns + 2)
        for column in range(columns):  # covSEard's derivative for a length scale
            spread = Numerics.squared_distances(inputs[:, [column]] / lengths[column])
            gradient[column] = np.sum(w * k_signal * spread) / 2
        gradient[columns] = np.sum(w * 2 * k_signal) / 2           # the signal's size
        gradient[columns + 1] = np.trace(w) * 2 * noise_variance / 2  # covNoise
        return float(value), gradient

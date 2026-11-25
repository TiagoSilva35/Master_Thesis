import numpy as np

from src.baselines.cdnod.gaussian_process import GaussianProcessWidths
from src.baselines.cdnod.numerics import Numerics


class ChangeDependence:
    """infer_nonsta_dir.m: how dependent the changes of P(causes) and of
    P(effect | causes) look across C, for one hypothesised direction. Each
    distribution is embedded, for every value of C, as a point in a kernel space;
    the score is the HSIC between the two sets of points. Under the Independent
    Change Principle the true direction gives the smaller score.

    Constants as in MATLAB: width 0.1 when none is given, width 1 on C, ridge 2.
    With a Gaussian process, the widths are learnt instead (its time input is
    1, ..., T, as in the original, while the kernels use C itself)."""

    def __init__(self, width: float, learn_widths: bool, widths: GaussianProcessWidths):
        self.width = width or 0.1
        self.learn_widths = learn_widths
        self.widths = widths

    def __call__(self, causes: np.ndarray, effect: np.ndarray, c_index: np.ndarray) -> float:
        rows = len(effect)
        x, y = Numerics.standardise(causes), Numerics.standardise(effect)
        theta = 1 / self.width ** 2
        conditional = self._conditional_embeddings(x, y, c_index, theta)
        marginal = self._marginal_embeddings(x, c_index, theta)
        h = Numerics.centring(rows)
        return float(np.sum((h @ conditional @ h).T * (h @ marginal @ h)) / rows ** 2)

    def _conditional_embeddings(self, x, y, c_index, theta) -> np.ndarray:
        """P(Y | X) for each C: Gaussian kernel between the embeddings."""
        rows, columns = x.shape
        kyy = Numerics.gaussian_kernel(y, y, theta)
        if self.learn_widths:
            time = np.arange(1, rows + 1, dtype=float)[:, None]
            start = [np.log(self.width)] * columns + [0.0, 0.0, np.log(np.sqrt(0.1))]  # log Wt = 0
            log_hyper = self.widths.fit(np.array(start), np.hstack([x, time]), self._targets(kyy))
            log_hyper[columns] = min(log_hyper[columns], np.log(1e4))  # a bound on C's width
            signal = float(np.exp(log_hyper[columns + 1]))
            kxt = Numerics.se_kernel(np.hstack([x, c_index]), np.exp(log_hyper[:columns + 1]), signal)
            inverse = Numerics.pd_inverse(kxt + np.exp(2 * log_hyper[-1]) * np.eye(rows))
            kxx = Numerics.se_kernel(x, np.exp(log_hyper[:columns]), signal)
            ktt = Numerics.se_kernel(c_index, np.exp(log_hyper[columns:columns + 1]), signal)
        else:
            kxx = Numerics.gaussian_kernel(x, x, theta)
            ktt = Numerics.gaussian_kernel(c_index, c_index, 1.0)
            inverse = Numerics.pd_inverse(kxx * ktt + 2 * np.eye(rows))
        embeddings = ktt @ ((kxx @ kxx @ kxx) * (inverse @ kyy @ inverse)) @ ktt / rows ** 2
        return self._gaussian_between(embeddings)

    def _marginal_embeddings(self, x, c_index, theta) -> np.ndarray:
        """P(X) for each C: Gaussian kernel between the embeddings."""
        rows = len(x)
        kxx = Numerics.gaussian_kernel(x, x, theta)
        if self.learn_widths:
            start = np.array([0.0, 0.0, np.log(np.sqrt(0.1))])
            log_hyper = self.widths.fit(start, c_index, self._targets(kxx))
            log_hyper[0] = min(log_hyper[0], np.log(1e4))
            ktt = Numerics.se_kernel(c_index, np.exp(log_hyper[:1]), float(np.exp(log_hyper[1])))
            inverse = Numerics.pd_inverse(ktt + np.exp(2 * log_hyper[-1]) * np.eye(rows))
        else:
            ktt = Numerics.gaussian_kernel(c_index, c_index, 1.0)
            inverse = Numerics.pd_inverse(ktt + 2 * np.eye(rows))
        return self._gaussian_between(ktt @ inverse @ kxx @ inverse @ ktt)

    @staticmethod
    def _targets(kernel: np.ndarray) -> np.ndarray:
        """The leading eigenvectors of a kernel matrix, scaled, as regression targets."""
        rows = len(kernel)
        values, vectors = Numerics.top_eigen((kernel + kernel.T) / 2, min(400, rows // 4))
        keep = values > values.max() * 1e-4
        values, vectors = values[keep], vectors[:, keep]
        return 2 * np.sqrt(rows) * vectors * np.sqrt(values) / np.sqrt(values[0])

    @staticmethod
    def _gaussian_between(embeddings: np.ndarray) -> np.ndarray:
        """Squared distances between embeddings from their inner products, then a
        Gaussian kernel whose width is the median distance."""
        diagonal = np.diag(embeddings)
        distances = diagonal[:, None] + diagonal[None, :] - 2 * embeddings
        median = np.median(distances[np.tril_indices(len(distances), -1)])
        return np.exp(-distances / median / 2)

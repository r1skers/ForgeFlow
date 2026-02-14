from statistics import mean, pstdev
from typing import Iterable

ResidualStats = dict[str, float]


class ResidualSigmaRule:
    def __init__(self, sigma_k: float = 3.0, min_sigma: float = 1e-9) -> None:
        if sigma_k <= 0:
            raise ValueError("sigma_k must be > 0")
        if min_sigma <= 0:
            raise ValueError("min_sigma must be > 0")

        self._sigma_k = float(sigma_k)
        self._min_sigma = float(min_sigma)
        self._mu = 0.0
        self._sigma = self._min_sigma
        self._fitted = False

    def fit(self, residuals: Iterable[float]) -> ResidualStats:
        residual_list = [float(value) for value in residuals]
        if not residual_list:
            raise ValueError("residuals must contain at least one value")

        mu = float(mean(residual_list))
        sigma = float(pstdev(residual_list))
        sigma = max(sigma, self._min_sigma)

        self._mu = mu
        self._sigma = sigma
        self._fitted = True
        return self.stats()

    def is_anomaly(self, residual: float) -> bool:
        if not self._fitted:
            raise RuntimeError("detector must be fitted before anomaly checks")
        deviation = abs(float(residual) - self._mu)
        return deviation > (self._sigma_k * self._sigma)

    def stats(self) -> ResidualStats:
        if not self._fitted:
            raise RuntimeError("detector must be fitted before reading stats")
        return {
            "mu": self._mu,
            "sigma": self._sigma,
            "sigma_k": self._sigma_k,
            "threshold": self._sigma_k * self._sigma,
        }

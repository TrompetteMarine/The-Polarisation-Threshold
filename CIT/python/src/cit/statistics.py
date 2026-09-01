from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

@dataclass(frozen=True)
class KernelSummary:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    susceptibility: float
    positive_susceptibility: float
    threshold: float
    positive_threshold: float
    support_95: float
    support_997: float

def summarize_kernel(times: np.ndarray, blocks: np.ndarray) -> KernelSummary:
    mean = blocks.mean(axis=1)
    se = blocks.std(axis=1, ddof=1) / np.sqrt(blocks.shape[1])
    z = norm.ppf(0.975)
    lower, upper = mean - z * se, mean + z * se
    susceptibility = float(np.trapezoid(mean, times))
    if susceptibility <= 0.0:
        raise ValueError("raw susceptibility must be positive")
    positive = np.maximum(mean, 0.0)
    positive_susceptibility = float(np.trapezoid(positive, times))
    if positive_susceptibility <= 0.0:
        raise ValueError("positive susceptibility must be non-zero")
    increments = np.r_[0.0, 0.5 * (positive[1:] + positive[:-1]) * np.diff(times)]
    share = np.cumsum(increments) / increments.sum()
    return KernelSummary(
        mean, lower, upper,
        susceptibility, positive_susceptibility,
        1.0 / susceptibility,
        1.0 / positive_susceptibility,
        float(times[np.argmax(share >= 0.95)]),
        float(times[np.argmax(share >= 0.997)]),
    )

def laplace_transform(times: np.ndarray, kernel: np.ndarray, arguments: np.ndarray) -> np.ndarray:
    return np.array([np.trapezoid(np.exp(-x * times) * kernel, times) for x in np.atleast_1d(arguments)])

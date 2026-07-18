from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .statistics import laplace_transform

def main_figure(response, summary, generator, output: Path) -> None:
    times = response.times
    kernel = summary.mean
    positive = np.maximum(kernel, 0.0)
    increments = np.r_[0.0, 0.5 * (positive[1:] + positive[:-1]) * np.diff(times)]
    cumulative = np.cumsum(increments)
    x = np.linspace(0.0, 2.2, 280)
    phi = laplace_transform(times, positive, x)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.7), dpi=220)
    ax = axes[0, 0]
    ax.hist(response.stationary_u, bins=110, range=(-3.3, 3.3), density=True, alpha=0.4)
    ax.axvspan(-1, 1, alpha=0.08)
    ax.axvline(-1, linestyle="--"); ax.axvline(1, linestyle="--")
    ax.axvline(-0.5, linestyle=":"); ax.axvline(0.5, linestyle=":")
    ax.set(title="(A) Stationary benchmark geometry", xlabel="Deviation u", ylabel="Density")

    ax = axes[0, 1]
    ax.fill_between(times, summary.lower, summary.upper, alpha=0.3, label="95% block band")
    ax.plot(times, kernel, linewidth=2.0, label="paired MC response")
    ax.axvline(summary.support_997, linestyle="--", label="99.7% support")
    ax.set(title="(B) Response kernel", xlabel="Time t", ylabel="k_h(t)")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(times, cumulative, linewidth=2.0)
    ax.axhline(summary.susceptibility, linestyle="--", label="MC susceptibility")
    ax.axhline(generator.susceptibility, linestyle=":", label="generator susceptibility")
    ax.set(title="(C) Cumulative susceptibility", xlabel="Time t", ylabel="integrated response")
    ax.legend()

    ax = axes[1, 1]
    for ratio in (0.75, 1.0, 1.25):
        kappa = ratio * summary.threshold
        ax.plot(x, 1 - kappa * phi, linewidth=2.0, label=f"{ratio:.2f} kappa*")
    ax.axhline(0, linewidth=0.8)
    ax.set(title="(D) Characteristic determinant", xlabel="x", ylabel="1-kappa Phi(x)")
    ax.legend()

    for ax in axes.ravel():
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

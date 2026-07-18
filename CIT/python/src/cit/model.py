from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ModelParameters:
    mean_reversion: float = 1.0
    diffusion: float = 1.0
    tolerance: float = 1.0
    reset_fraction: float = 0.5
    active_intensity: float = 2.0

    def validate(self) -> None:
        if min(self.mean_reversion, self.diffusion, self.tolerance) <= 0:
            raise ValueError("mean_reversion, diffusion, and tolerance must be positive")
        if not 0 < self.reset_fraction < 1:
            raise ValueError("reset_fraction must lie in (0,1)")
        if self.active_intensity < 0:
            raise ValueError("active_intensity must be non-negative")

@dataclass(frozen=True)
class SimulationParameters:
    particles: int = 60_000
    blocks: int = 60
    burn_steps: int = 3_000
    burn_dt: float = 0.01
    horizon: float = 8.0
    dt: float = 0.02
    delta: float = 0.02
    seed: int = 20_260_603

    def validate(self) -> None:
        if self.particles <= 0 or self.blocks <= 1 or self.particles < self.blocks:
            raise ValueError("invalid particle/block configuration")
        if min(self.burn_steps, self.burn_dt, self.horizon, self.dt, self.delta) <= 0:
            raise ValueError("time and finite-difference settings must be positive")

def one_step(u: np.ndarray, duration: np.ndarray, normal: np.ndarray, uniform: np.ndarray, dt: float, model: ModelParameters) -> None:
    u += -model.mean_reversion * u * dt + model.diffusion * np.sqrt(dt) * normal
    outside = np.abs(u) > model.tolerance
    duration += outside * dt
    reset = outside & (uniform < model.active_intensity * dt)
    u[reset] = np.sign(u[reset]) * model.reset_fraction * model.tolerance
    duration[reset] = 0.0

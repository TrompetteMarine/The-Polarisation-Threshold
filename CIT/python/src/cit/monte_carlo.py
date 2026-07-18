from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .model import ModelParameters, SimulationParameters, one_step

@dataclass(frozen=True)
class ResponseResult:
    times: np.ndarray
    block_responses: np.ndarray
    stationary_u: np.ndarray
    stationary_duration: np.ndarray

    @property
    def mean_response(self) -> np.ndarray:
        return self.block_responses.mean(axis=1)

def simulate_stationary(model: ModelParameters, simulation: SimulationParameters):
    model.validate(); simulation.validate()
    rng = np.random.default_rng(simulation.seed)
    u = rng.normal(0.0, 1.3, size=simulation.particles)
    duration = np.zeros(simulation.particles)
    for _ in range(simulation.burn_steps):
        one_step(u, duration, rng.normal(size=simulation.particles), rng.random(simulation.particles), simulation.burn_dt, model)
    return u, duration, rng

def paired_response(model: ModelParameters, simulation: SimulationParameters) -> ResponseResult:
    u, duration, rng = simulate_stationary(model, simulation)
    block_size = simulation.particles // simulation.blocks
    n = block_size * simulation.blocks
    u = u[:n].copy(); duration = duration[:n].copy()
    u_plus = u + simulation.delta; u_minus = u - simulation.delta
    a_plus = duration.copy(); a_minus = duration.copy()
    steps = int(round(simulation.horizon / simulation.dt))
    blocks = np.empty((steps + 1, simulation.blocks))
    diff = (u_plus - u_minus) / (2 * simulation.delta)
    blocks[0] = diff.reshape(simulation.blocks, block_size).mean(axis=1)
    for step in range(1, steps + 1):
        normal = rng.normal(size=n); uniform = rng.random(n)
        one_step(u_plus, a_plus, normal, uniform, simulation.dt, model)
        one_step(u_minus, a_minus, normal, uniform, simulation.dt, model)
        diff = (u_plus - u_minus) / (2 * simulation.delta)
        blocks[step] = diff.reshape(simulation.blocks, block_size).mean(axis=1)
    times = np.linspace(0.0, simulation.horizon, steps + 1)
    return ResponseResult(times, blocks, u, duration)

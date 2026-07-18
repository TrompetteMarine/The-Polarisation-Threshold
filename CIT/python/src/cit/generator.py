from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from .model import ModelParameters

@dataclass(frozen=True)
class GeneratorResult:
    grid: np.ndarray
    stationary_mass: np.ndarray
    response_direction: np.ndarray
    poisson_solution: np.ndarray
    susceptibility: float
    threshold: float

def build_generator(model: ModelParameters, grid_points: int = 801, domain: float = 5.0) -> tuple[np.ndarray, csr_matrix]:
    model.validate()
    if grid_points < 51 or grid_points % 2 == 0:
        raise ValueError("grid_points must be odd and at least 51")
    grid = np.linspace(-domain, domain, grid_points)
    dx = grid[1] - grid[0]
    diffusion = 0.5 * model.diffusion**2
    generator = lil_matrix((grid_points, grid_points))
    for i, u in enumerate(grid):
        drift = -model.mean_reversion * u
        rate_plus = diffusion / dx**2 + max(drift, 0.0) / dx
        rate_minus = diffusion / dx**2 + max(-drift, 0.0) / dx
        generator[i, i + 1 if i < grid_points - 1 else i - 1] += rate_plus
        generator[i, i - 1 if i > 0 else i + 1] += rate_minus
        if abs(u) > model.tolerance:
            target = np.sign(u) * model.reset_fraction * model.tolerance
            target_index = int(np.argmin(np.abs(grid - target)))
            if target_index != i:
                generator[i, target_index] += model.active_intensity
        generator[i, i] = -generator[i].sum()
    return grid, generator.tocsr()

def generator_cross_check(model: ModelParameters, grid_points: int = 801, domain: float = 5.0) -> GeneratorResult:
    grid, generator = build_generator(model, grid_points, domain)
    n = generator.shape[0]
    system = generator.T.tolil(); rhs = np.zeros(n)
    system[-1, :] = 1.0; rhs[-1] = 1.0
    mass = np.maximum(spsolve(system.tocsr(), rhs), 0.0); mass /= mass.sum()
    dx = grid[1] - grid[0]
    direction = np.empty_like(mass)
    direction[1:-1] = -(mass[2:] - mass[:-2]) / (2 * dx)
    direction[0] = -(mass[1] - mass[0]) / dx
    direction[-1] = -(mass[-1] - mass[-2]) / dx
    direction -= mass * direction.sum(); direction /= float(grid @ direction)
    system = (-generator.T).tolil(); rhs = direction.copy()
    system[-1, :] = 1.0; rhs[-1] = 0.0
    poisson = spsolve(system.tocsr(), rhs)
    susceptibility = float(grid @ poisson)
    return GeneratorResult(grid, mass, direction, poisson, susceptibility, 1.0 / susceptibility)

from pathlib import Path
import numpy as np
from cit.generator import generator_cross_check
from cit.model import ModelParameters, SimulationParameters
from cit.monte_carlo import paired_response
from cit.pipeline import run_pipeline
from cit.statistics import laplace_transform, summarize_kernel

def test_response_normalization_and_centering() -> None:
    simulation = SimulationParameters(particles=2400, blocks=12, burn_steps=250, horizon=1.0, dt=0.05)
    result = paired_response(ModelParameters(), simulation)
    assert np.allclose(result.block_responses[0], 1.0, atol=1e-12)
    assert abs(result.stationary_u.mean()) < 0.08

def test_exponential_kernel_contract() -> None:
    times = np.linspace(0, 12, 1201)
    kernel = np.exp(-times)
    blocks = np.tile(kernel[:, None], (1, 8))
    summary = summarize_kernel(times, blocks)
    assert abs(summary.susceptibility - 1.0) < 2e-3
    assert abs(summary.threshold - 1.0) < 2e-3
    values = laplace_transform(times, kernel, np.array([0.0, 1.0]))
    assert abs(values[0] - 1.0) < 2e-3
    assert abs(values[1] - 0.5) < 2e-3

def test_generator_contract() -> None:
    result = generator_cross_check(ModelParameters(), grid_points=401, domain=5.0)
    assert abs(result.stationary_mass.sum() - 1.0) < 1e-10
    assert abs(result.grid @ result.response_direction - 1.0) < 1e-10
    assert 0.63 < result.susceptibility < 0.72
    assert 1.38 < result.threshold < 1.59

def test_fast_pipeline_writes_outputs(tmp_path: Path) -> None:
    report = run_pipeline(tmp_path, publication=False)
    assert (tmp_path / "response_kernel.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "figure_numerical.png").exists()
    assert abs(report["k0"] - 1.0) < 1e-12

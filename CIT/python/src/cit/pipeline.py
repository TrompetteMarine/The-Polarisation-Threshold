from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import pandas as pd
from .figures import main_figure
from .generator import generator_cross_check
from .model import ModelParameters, SimulationParameters
from .monte_carlo import paired_response
from .statistics import summarize_kernel

def run_pipeline(output_dir: Path, publication: bool = False) -> dict[str, float]:
    model = ModelParameters()
    simulation = SimulationParameters() if publication else SimulationParameters(
        particles=12_000, blocks=24, burn_steps=700, burn_dt=0.01,
        horizon=5.0, dt=0.025, delta=0.02,
    )
    response = paired_response(model, simulation)
    summary = summarize_kernel(response.times, response.block_responses)
    generator = generator_cross_check(model, grid_points=801 if publication else 401)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "t": response.times,
        "k_hat": summary.mean,
        "lower95": summary.lower,
        "upper95": summary.upper,
    }).to_csv(output_dir / "response_kernel.csv", index=False)
    main_figure(response, summary, generator, output_dir / "figure_numerical.png")
    report = {
        "mc_susceptibility": summary.susceptibility,
        "mc_positive_susceptibility": summary.positive_susceptibility,
        "mc_threshold": summary.threshold,
        "mc_positive_threshold": summary.positive_threshold,
        "generator_susceptibility": generator.susceptibility,
        "generator_threshold": generator.threshold,
        "k0": float(summary.mean[0]),
        "support_95": summary.support_95,
        "support_997": summary.support_997,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "configuration.json").write_text(json.dumps({
        "model": asdict(model), "simulation": asdict(simulation)
    }, indent=2) + "\n")
    return report

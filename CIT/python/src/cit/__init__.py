"""Numerical replication package for the Consensus Instability Theorem."""

from .model import ModelParameters, SimulationParameters
from .monte_carlo import paired_response
from .statistics import summarize_kernel, laplace_transform
from .generator import generator_cross_check
from .pipeline import run_pipeline

__all__ = [
    "ModelParameters",
    "SimulationParameters",
    "paired_response",
    "summarize_kernel",
    "laplace_transform",
    "generator_cross_check",
    "run_pipeline",
]

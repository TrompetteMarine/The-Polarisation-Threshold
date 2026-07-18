# Consensus Instability Theorem — Independent Replication Project

This directory is an independent, code-only replication project for the numerical results accompanying **The Consensus Instability Theorem**. It is deliberately isolated from the parent `BeliefSim.jl` package and does not import the polarisation-threshold implementation.

Two implementations are supplied:

- `python/`: reference implementation used for the archived numerical checks and publication figures;
- `julia/`: independent Julia implementation of the same active-hazard benchmark, paired response experiment, finite-state generator cross-check, susceptibility calculation, and threshold diagnostics.

Neither implementation contains the manuscript, LaTeX source, bibliography, or formal-proof artefacts.

## Canonical benchmark

- mean reversion `λ = 1`;
- diffusion `σ = 1`;
- tolerance `Θ = 1`;
- reset fraction `c₀ = 0.5`;
- active completion intensity `ν₀ = 2`;
- seed `20260603`.

Expected publication-scale values are approximately `Φ̂(0)=0.667`, generator `Φ(0)=0.668–0.670`, `κ*=1.49`, and exact `k(0)=1`.

## Python

```bash
cd CIT/python
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
cit-run --fast
```

## Julia

```bash
cd CIT/julia
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); Pkg.test()'
julia --project=. scripts/run_all.jl --fast
```

Publication run:

```bash
julia --project=. scripts/run_all.jl --publication
```

## Independence contract

The CIT code may share benchmark parameters and notation with the parent repository, but it must not depend on `BeliefSim`, its internal modules, or its project environment.

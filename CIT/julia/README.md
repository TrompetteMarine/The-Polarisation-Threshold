# CITReplication.jl

Independent Julia implementation of the numerical replication for the Consensus Instability Theorem.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); Pkg.test()'
julia --project=. scripts/run_all.jl --fast
julia --project=. scripts/run_all.jl --publication
```

The package reproduces the active-hazard particle system, common-random-number paired response kernel, integrated susceptibility, effective-support diagnostics, finite-state generator / Poisson cross-check, critical threshold, and four-panel numerical figure.

It has no dependency on the parent `BeliefSim.jl` package.

.PHONY: figs setup test clean help fig5 fig6 figs_abc

# Default target
help:
	@echo "Available targets:"
	@echo "  setup  - Install dependencies and prepare environment"
	@echo "  figs   - Generate all figures (requires BifurcationKit)"
	@echo "  fig5   - Generate phase diagram κ*(c0, σ)"
	@echo "  fig6   - Generate density snapshots below/above κ*"
	@echo "  figs_abc - Generate Fig A (YAML), Fig B, and Fig C"
	@echo "  test   - Run test suite"
	@echo "  clean  - Remove generated files"

setup:
	./scripts/setup_environment.sh

figs:
	julia --project=. scripts/make_phase_portraits.jl
	julia --project=. scripts/scan_hopf_and_cycles.jl
	julia --project=. scripts/scan_homoclinic.jl

fig5:
	julia --project=. scripts/fig5_phase_kstar.jl

fig6:
	julia --project=. scripts/fig6_density_snapshots.jl

figs_abc:
	julia --project=. scripts/analyze_from_yaml.jl configs/figA_zoom.yaml
	julia --project=. scripts/fig5_phase_kstar.jl
	julia --project=. scripts/fig6_density_snapshots.jl

test:
	julia --project=. -e 'using Pkg; Pkg.test()'

clean:
	rm -rf figs/*.png figs/*.pdf outputs/*
	find . -name "*.jl.cov" -delete
	find . -name "*.jl.*.cov" -delete
	find . -name "*.jl.mem" -delete

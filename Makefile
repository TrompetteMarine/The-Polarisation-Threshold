.PHONY: figs setup test clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  setup  - Install dependencies and prepare environment"
	@echo "  figs   - Generate all figures (requires BifurcationKit)"
	@echo "  test   - Run test suite"
	@echo "  clean  - Remove generated files"

setup:
	./scripts/setup_environment.sh

figs:
	julia --project=. scripts/make_phase_portraits.jl
	julia --project=. scripts/scan_hopf_and_cycles.jl
	julia --project=. scripts/scan_homoclinic.jl

test:
	julia --project=. -e 'using Pkg; Pkg.test()'

clean:
	rm -rf figs/*.png figs/*.pdf outputs/*
	find . -name "*.jl.cov" -delete
	find . -name "*.jl.*.cov" -delete
	find . -name "*.jl.mem" -delete
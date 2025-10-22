"""
Test the YAML analysis script
"""

include("analyze_from_yaml.jl")

@info "Testing YAML analysis script..."

# Test with example config
config_path = joinpath(@__DIR__, "..", "configs", "example_sweep.yaml")

if !isfile(config_path)
    @error "Example config not found: $config_path"
    @error "Create it first!"
    exit(1)
end

@info "Running analysis on: $config_path"
results = analyze_config(config_path)

@info ""
@info "Test complete!"
@info "Results: $results"
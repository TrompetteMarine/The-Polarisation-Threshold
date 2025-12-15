#!/usr/bin/env julia
# Thin wrapper so users can run the corrected welfare analysis from the spec.
# We reuse the implementation in `run_welfare_contours.jl`.

include("run_welfare_contours.jl")

# Defer to the main defined in run_welfare_contours.jl
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


push!(LOAD_PATH, joinpath(@__DIR__, "..", "src", "bifurcation"))

using Logging

try
    using ModelInterface
    using BifurcationCore
catch e
    @error "Failed to load bifurcation modules" exception=e
    exit(1)
end

function main()
    p = ModelInterface.default_params()
    u_s = zeros(2)

    @info "Scanning for homoclinic orbits..."
    found = false
    for κ in 0.9:0.001:1.3
        pκ = ModelInterface.kappa_set(p, κ)
        if BifurcationCore.find_homoclinic_near_saddle(u_s, pκ; tmax=2000.0, tol=1e-2)
            @info "Possible homoclinic at κ ≈ $κ"
            found = true
            break
        end
    end

    if !found
        @info "No homoclinic candidate detected in scanned window."
    end
end

main()
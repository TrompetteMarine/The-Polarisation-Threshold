push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Logging
using ModelInterface
using Bifurcation

p = ModelInterface.default_params()
u_s = zeros(2)

found = false
for κ in 0.9:0.001:1.3
    pκ = ModelInterface.kappa_set(p, κ)
    if Bifurcation.find_homoclinic_near_saddle(u_s, pκ; tmax=2000.0, tol=1e-2)
        @info "Possible homoclinic at κ ≈ $κ"
        found = true
        break
    end
end

if !found
    @info "No homoclinic candidate detected in scanned window."
end

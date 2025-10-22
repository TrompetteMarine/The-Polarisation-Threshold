include("../src/bifurcation/model_interface.jl")
include("../src/bifurcation/simple_continuation.jl")

using .ModelInterface
using .SimpleContinuation
using LinearAlgebra

@info "Testing ModelInterface..."

# Get default parameters
p = ModelInterface.default_params()
@info "Default parameters:" p

# Test function evaluation
u = zeros(2)
F = ModelInterface.f(u, p)
@info "f([0,0]) =" F

# Test jacobian
J = ModelInterface.jacobian(u, p)
@info "J([0,0]) =" J

# Test different κ values
for κ in [0.8, 1.0, 1.2]
    p_test = ModelInterface.kappa_set(p, κ)
    @info "\nTesting κ = $κ"
    @info "Parameters:" p_test
    
    try
        u_eq = SimpleContinuation.newton_solve(
            ModelInterface.f,
            ModelInterface.jacobian,
            zeros(2),
            p_test
        )
        
        F_check = ModelInterface.f(u_eq, p_test)
        @info "  Equilibrium:" u_eq
        @info "  Residual norm:" norm(F_check)
        
        J = ModelInterface.jacobian(u_eq, p_test)
        λs = eigvals(J)
        @info "  Eigenvalues:" λs
    catch e
        @error "  Failed!" exception=e
    end
end
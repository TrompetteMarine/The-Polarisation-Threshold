using Test
using BeliefSim
using BeliefSim.OUResets
using BeliefSim.Types

# Small helper to find argmin while ignoring NaNs
function find_argmin(surface, theta_grid, c0_grid)
    finite_indices = findall(isfinite, surface)
    @test !isempty(finite_indices) "Surface has no finite entries"
    vals = map(idx -> surface[idx], finite_indices)
    _, pos = findmin(vals)
    i, j = Tuple(finite_indices[pos])
    return theta_grid[i], c0_grid[j], i, j
end

@testset "Mean-field welfare split" begin
    params_base = Params(λ = 0.65, σ = 1.15, Θ = 0.9, c0 = 0.5, hazard = StepHazard(0.5))

    theta_grid = collect(range(0.7, 1.1; length = 3))
    c0_grid    = collect(range(0.35, 0.65; length = 3))

    V_surface   = fill(NaN, length(theta_grid), length(c0_grid))
    L_dec       = similar(V_surface)
    L_soc       = similar(V_surface)
    Phi_surface = similar(V_surface)

    αV = 1.0
    K_reset = 0.5
    c_pol = 2.0
    φA = 0.1
    κ_ratio_max = 2.0

    for (i, theta) in enumerate(theta_grid)
        for (j, c0) in enumerate(c0_grid)
            V = compute_stationary_variance(
                theta, c0, params_base;
                N       = 800,
                T       = 80.0,
                dt      = 0.02,
                burn_in = 20.0,
                seed    = 11 * i + j,
            )
            V_surface[i, j] = V

            if isfinite(V) && V > 0
                p_ij = Params(λ = params_base.λ, σ = params_base.σ, Θ = theta, c0 = c0, hazard = params_base.hazard)

                L_dec[i, j] = welfare_loss(V, theta, p_ij; regime = :dec, αV = αV, K_reset = K_reset)

                λ10, λ1_dot = compute_lambda1_and_derivative(V, p_ij; δκ = 0.02, L = 4.0, M = 151)
                Phi_surface[i, j] = bifurcation_loss(
                    V, p_ij;
                    c_pol       = c_pol,
                    φA          = φA,
                    κ_ratio_max = κ_ratio_max,
                    δκ          = 0.02,
                    L           = 4.0,
                    M           = 151,
                    lambda10    = λ10,
                    lambda1_dot = λ1_dot,
                )

                L_soc[i, j] = welfare_loss(
                    V, theta, p_ij;
                    regime      = :soc,
                    αV          = αV,
                    K_reset     = K_reset,
                    c_pol       = c_pol,
                    φA          = φA,
                    κ_ratio_max = κ_ratio_max,
                    δκ          = 0.02,
                    L           = 4.0,
                    M           = 151,
                    Phi         = Phi_surface[i, j],
                    lambda10    = λ10,
                    lambda1_dot = λ1_dot,
                )
            end
        end
    end

    @test any(isfinite, L_dec) "Decentralised surface has no finite entries"
    @test any(isfinite, L_soc) "Planner surface has no finite entries"
    @test all(x -> x >= -1e-8, Phi_surface[isfinite.(Phi_surface)]) "Phi should be non-negative up to numerical noise"

    theta_dec, c0_dec, idec, jdec = find_argmin(L_dec, theta_grid, c0_grid)
    theta_soc, c0_soc, isoc, jsoc = find_argmin(L_soc, theta_grid, c0_grid)

    V_dec = V_surface[idec, jdec]
    V_soc = V_surface[isoc, jsoc]

    @test theta_soc <= theta_dec
    @test V_soc <= V_dec

    @info "Decentralised optimum" theta_dec=theta_dec c0_dec=c0_dec V_dec=V_dec
    @info "Social optimum" theta_soc=theta_soc c0_soc=c0_soc V_soc=V_soc
end


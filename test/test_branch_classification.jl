using Test
using Random

include(joinpath(@__DIR__, "..", "scripts", "modules", "BranchClassification.jl"))
using .BranchClassification

@testset "BranchClassification" begin
    rng = MersenneTwister(1)
    n_runs = 50
    t = collect(0.0:1.0:100.0)
    mean_traj = randn(rng, n_runs, length(t)) .* 0.01
    signs, mean_late, decided, tau_used, decided_share, plus_share, minus_share, undecided =
        compute_branch_signs(mean_traj, t)
    @test decided_share <= 1.0
    @test abs(plus_share + minus_share - 1.0) < 1e-6 || decided_share == 0.0
end

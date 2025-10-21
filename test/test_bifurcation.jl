push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Test
using ModelInterface

@testset "Jacobian sanity" begin
    p = ModelInterface.default_params()
    u = zeros(2)
    J = ModelInterface.jacobian(u, p)
    @test size(J, 1) == size(J, 2)
    @test size(J, 1) >= 2
end

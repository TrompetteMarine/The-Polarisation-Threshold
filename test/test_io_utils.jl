using Test
using DataFrames

include(joinpath(@__DIR__, "..", "scripts", "modules", "IOUtils.jl"))
using .IOUtils

@testset "IOUtils" begin
    df = DataFrame(alpha=[1, 2], beta_hat=[0.5, 0.6])
    @test has_col(df, :alpha) == true
    @test has_col(df, :beta_hat) == true
    @test has_col(df, :nonexistent) == false
    @test has_col(DataFrame(), :x) == false

    @test get_col(df, :alpha) == df.alpha
    @test pick_col(df, [:missing, :beta_hat]) == df.beta_hat

    @test sanitize_json(Dict("x" => NaN)) == Dict("x" => nothing)
end

using Test

@testset "Numerical validation (smoke)" begin
    if get(ENV, "RUN_NUMERICAL_VALIDATION_SMOKE", "0") == "1"
        cmd = `julia --project=. scripts/numerical_validation/run_validation.jl --config configs/validation.yaml --quick`
        run(cmd)
        @test isfile("artifacts/numerical_validation/robustness_table.csv")
        @test isfile("artifacts/numerical_validation/robustness_table.tex")
        @test isfile("artifacts/numerical_validation/run_manifest.json")
        @test isfile("artifacts/numerical_validation/run.log")
    else
        @info "Skipping numerical validation smoke test; set RUN_NUMERICAL_VALIDATION_SMOKE=1 to enable."
        @test true
    end
end

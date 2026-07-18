module KernelStatistics

using Statistics

struct KernelSummary
    mean::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    susceptibility::Float64
    positive_susceptibility::Float64
    threshold::Float64
    support_95::Float64
    support_997::Float64
end

trapz(t, y) = sum((y[1:end-1] .+ y[2:end]) .* diff(t)) / 2

function summarize_kernel(times::Vector{Float64}, blocks::Matrix{Float64}; alpha=0.05)
    alpha == 0.05 || throw(ArgumentError("the stdlib implementation currently supports alpha=0.05"))
    μ = vec(mean(blocks; dims=2))
    se = vec(std(blocks; dims=2, corrected=true)) ./ sqrt(size(blocks, 2))
    z = 1.959963984540054
    lower, upper = μ .- z .* se, μ .+ z .* se
    susceptibility = trapz(times, μ)
    positive = max.(μ, 0.0)
    positive_susceptibility = trapz(times, positive)
    positive_susceptibility > 0 || throw(ArgumentError("positive susceptibility must be non-zero"))
    increments = vcat(0.0, (positive[1:end-1] .+ positive[2:end]) .* diff(times) ./ 2)
    share = cumsum(increments) ./ sum(increments)
    i95 = findfirst(>=(0.95), share)
    i997 = findfirst(>=(0.997), share)
    KernelSummary(μ, lower, upper, susceptibility, positive_susceptibility,
                  inv(positive_susceptibility), times[i95], times[i997])
end

laplace_transform(times, kernel, args) = [trapz(times, exp.(-x .* times) .* kernel) for x in args]

end

module CITPlotting

using Plots
using ..MonteCarlo: ResponseResult
using ..KernelStatistics: KernelSummary, laplace_transform
using ..GeneratorCheck: GeneratorResult

function main_figure(response::ResponseResult, summary::KernelSummary, generator::GeneratorResult, output::AbstractString)
    times = response.times
    kernel = summary.mean
    positive = max.(kernel, 0.0)
    arguments = collect(range(0, 2.2; length=280))
    phi = laplace_transform(times, positive, arguments)
    cumulative = cumsum(vcat(0.0, (positive[1:end-1] .+ positive[2:end]) .* diff(times) ./ 2))

    p1 = histogram(response.stationary_u; bins=110, normalize=:pdf, alpha=0.35,
                   label="particle histogram", xlabel="u", ylabel="density",
                   title="(A) Stationary benchmark")
    vspan!(p1, [-1, 1]; alpha=0.08, label="tolerance band")
    vline!(p1, [-1, 1]; linestyle=:dash, label="±Θ")
    vline!(p1, [-0.5, 0.5]; linestyle=:dot, label="±c₀Θ")

    p2 = plot(times, kernel; ribbon=(kernel-summary.lower, summary.upper-kernel),
              label="paired MC response", xlabel="t", ylabel="kₕ(t)",
              title="(B) Response kernel")
    vline!(p2, [summary.support_997]; linestyle=:dash, label="99.7% support")

    p3 = plot(times, cumulative; label="MC cumulative susceptibility", xlabel="t",
              ylabel="∫₀ᵗ kₕ(s)ds", title="(C) Cumulative susceptibility")
    hline!(p3, [summary.susceptibility]; linestyle=:dash, label="MC Φₕ(0)")
    hline!(p3, [generator.susceptibility]; linestyle=:dot, label="generator Φₕ(0)")

    p4 = plot(; xlabel="x", ylabel="1-κΦₕ(x)", title="(D) Characteristic determinant")
    for κ in (0.75*summary.threshold, summary.threshold, 1.25*summary.threshold)
        plot!(p4, arguments, 1 .- κ .* phi; label="κ=$(round(κ/summary.threshold,digits=2))κ*")
    end
    hline!(p4, [0.0]; color=:black, label="")

    figure = plot(p1, p2, p3, p4; layout=(2,2), size=(1200,850), dpi=180)
    mkpath(dirname(output))
    savefig(figure, output)
    output
end

end

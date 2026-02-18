module PlotGrammar

export GRAMMAR,
       apply_plot_grammar!,
       style_axis!,
       add_time_colorbar!,
       add_style_legend!,
       time_color

const GRAMMAR = (
    fontsize = 13,
    titlefontsize = 15,
    labelsize = 13,
    ticklabelsize = 11,
    linewidth_main = 2.6,
    linewidth_secondary = 1.2,
    alpha_secondary = 0.35,
    ci_alpha = 0.12,
    gridalpha = 0.15,
    colormap_time = :viridis,
)

function apply_plot_grammar!(mk::Module)
    theme = Base.invokelatest(
        mk.Theme;
        fontsize = GRAMMAR.fontsize,
        Axis = (
            xlabelsize = GRAMMAR.labelsize,
            ylabelsize = GRAMMAR.labelsize,
            titlesize = GRAMMAR.titlefontsize,
            xticklabelsize = GRAMMAR.ticklabelsize,
            yticklabelsize = GRAMMAR.ticklabelsize,
            xgridvisible = true,
            ygridvisible = true,
            xgridcolor = (:black, GRAMMAR.gridalpha),
            ygridcolor = (:black, GRAMMAR.gridalpha),
            framevisible = false,
        ),
        Legend = (
            framevisible = false,
            labelsize = GRAMMAR.ticklabelsize,
            patchsize = (24, 12),
        ),
    )
    Base.invokelatest(mk.set_theme!, theme)
    return GRAMMAR
end

function style_axis!(mk::Module, ax; xlabel=nothing, ylabel=nothing, title=nothing)
    if xlabel !== nothing
        ax.xlabel = xlabel
    end
    if ylabel !== nothing
        ax.ylabel = ylabel
    end
    if title !== nothing
        ax.title = title
    end
    ax.xgridvisible = true
    ax.ygridvisible = true
    ax.xgridcolor = (:black, GRAMMAR.gridalpha)
    ax.ygridcolor = (:black, GRAMMAR.gridalpha)
    ax.framevisible = false
    return ax
end

function _default_time_ticks(tmin::Real, tmax::Real)
    if !isfinite(tmin) || !isfinite(tmax) || tmin == tmax
        return [tmin]
    end
    return collect(range(tmin, tmax; length=5))
end

function add_time_colorbar!(
    mk::Module,
    slot;
    tmin::Real,
    tmax::Real,
    colormap = GRAMMAR.colormap_time,
    ticks = nothing,
    label::AbstractString = "time t",
)
    t_ticks = ticks === nothing ? _default_time_ticks(tmin, tmax) : ticks
    return mk.Colorbar(slot, colormap=colormap, colorrange=(tmin, tmax), ticks=t_ticks, label=label)
end

function add_style_legend!(
    mk::Module,
    slot;
    include_plus_minus::Bool = false,
    scenario_entries::Vector{Tuple{String, Any}} = Tuple{String, Any}[],
)
    elements = mk.LineElement[
        mk.LineElement(color=:black, linestyle=:solid, linewidth=GRAMMAR.linewidth_main),
        mk.LineElement(color=:black, linestyle=:dash, linewidth=GRAMMAR.linewidth_secondary),
    ]
    labels = String["aligned", "mixture"]

    if include_plus_minus
        push!(elements, mk.LineElement(color=:gray, linestyle=:dot, linewidth=GRAMMAR.linewidth_secondary))
        push!(elements, mk.LineElement(color=:gray, linestyle=:dashdot, linewidth=GRAMMAR.linewidth_secondary))
        push!(labels, "plus", "minus")
    end

    for (label, color) in scenario_entries
        push!(elements, mk.LineElement(color=color, linestyle=:solid, linewidth=GRAMMAR.linewidth_main))
        push!(labels, label)
    end

    return mk.Legend(slot, elements, labels; orientation=:horizontal, framevisible=false, tellwidth=false)
end

function time_color(
    mk::Module,
    tval::Real,
    tmin::Real,
    tmax::Real;
    colormap = GRAMMAR.colormap_time,
)
    cmap = mk.cgrad(colormap, 256, categorical=false)
    if !isfinite(tval) || tmax == tmin
        return cmap[end]
    end
    α = clamp((tval - tmin) / (tmax - tmin), 0.0, 1.0)
    idx = clamp(Int(round(1 + α * (length(cmap) - 1))), 1, length(cmap))
    return cmap[idx]
end

end # module

#!/usr/bin/env julia
# ============================================================
# eigen_kappa_postprocess_flags.jl
#
# Post-process kappa sweep CSVs to flag clustered crossings
# and overlap drops (branch switching).
#
# Usage:
#   julia --project=. scripts/eigen_kappa_postprocess_flags.jl
# ============================================================

using DelimitedFiles
using Printf
using Dates

# ----------------------------
# Configuration
# ----------------------------
const IN_ROOT = joinpath("figs", "eigen_diag", "validity")
const OUT_PATH = joinpath(IN_ROOT, "branch_switch_flags.csv")
const FILE_PREFIX = "lambda_curve_"
const FILE_SUFFIX = ".csv"

const CROSSING_CLUSTER_DELTA = 0.2
const OVERLAP_ABS_THRESH = 0.2
const OVERLAP_REL_DROP = 0.5
const MAX_LIST = 8

# ----------------------------
# Utilities
# ----------------------------
parse_float(s::AbstractString) = (v = tryparse(Float64, s); v === nothing ? NaN : v)

function read_columns(path::AbstractString, cols::Vector{String})
    lines = readlines(path)
    if isempty(lines)
        return Dict(c => Float64[] for c in cols)
    end
    header = split(chomp(lines[1]), ',')
    idx = Dict{String, Int}()
    for c in cols
        j = findfirst(==(c), header)
        idx[c] = j === nothing ? 0 : j
    end
    data = Dict(c => Float64[] for c in cols)
    for line in lines[2:end]
        isempty(line) && continue
        parts = split(chomp(line), ',')
        for c in cols
            j = idx[c]
            val = (j == 0 || j > length(parts)) ? NaN : parse_float(parts[j])
            push!(data[c], val)
        end
    end
    return data
end

function find_crossings(kappa::Vector{Float64}, lambda1::Vector{Float64})
    xs = Float64[]
    n = length(kappa)
    for i in 1:(n - 1)
        a, b = lambda1[i], lambda1[i + 1]
        if !isfinite(a) || !isfinite(b)
            continue
        end
        if a == 0.0
            push!(xs, kappa[i])
        elseif (a < 0 && b > 0) || (a > 0 && b < 0)
            w = -a / (b - a)
            push!(xs, kappa[i] + w * (kappa[i + 1] - kappa[i]))
        end
    end
    sort!(xs)
    return xs
end

function cluster_crossings(xs::Vector{Float64}; delta::Float64)
    clusters = Tuple{Float64, Float64, Int}[]
    n = length(xs)
    i = 1
    while i <= n
        j = i
        while j < n && (xs[j + 1] - xs[j]) <= delta
            j += 1
        end
        if j > i
            push!(clusters, (xs[i], xs[j], j - i + 1))
        end
        i = j + 1
    end
    return clusters
end

function format_list(vals::Vector{Float64}; max_items::Int=MAX_LIST)
    if isempty(vals)
        return ""
    end
    n = min(length(vals), max_items)
    parts = [@sprintf("%.4f", vals[i]) for i in 1:n]
    suffix = length(vals) > max_items ? "..." : ""
    return join(parts, ";") * suffix
end

function overlap_drop_stats(kappa::Vector{Float64}, overlap::Vector{Float64};
                            abs_thresh::Float64, rel_drop::Float64)
    idx = findall(isfinite, overlap)
    if isempty(idx)
        return (min_overlap=NaN, abs_idx=Int[], rel_idx=Int[], available=false)
    end
    min_overlap = minimum(overlap[idx])

    abs_idx = Int[]
    for i in idx
        if overlap[i] < abs_thresh
            push!(abs_idx, i)
        end
    end

    rel_idx = Int[]
    for j in 2:length(idx)
        i_prev = idx[j - 1]
        i_cur = idx[j]
        if overlap[i_prev] > 0 && overlap[i_cur] / overlap[i_prev] < rel_drop
            push!(rel_idx, i_cur)
        end
    end

    return (min_overlap=min_overlap, abs_idx=abs_idx, rel_idx=rel_idx, available=true)
end

function parse_tag(path::AbstractString)
    name = split(path, '/')[end]
    m = match(r"lambda_curve_h([^_]+)_L([^_]+)_M(\d+)\.csv$", name)
    if m === nothing
        return (nu0=NaN, L=NaN, M=NaN)
    end
    nu0 = parse_float(replace(m.captures[1], "p" => "."))
    L = parse_float(replace(m.captures[2], "p" => "."))
    M = parse_float(m.captures[3])
    return (nu0=nu0, L=L, M=M)
end

# ----------------------------
# Main
# ----------------------------
function main()
    files = filter(f -> startswith(f, FILE_PREFIX) && endswith(f, FILE_SUFFIX),
                   readdir(IN_ROOT; join=true))
    rows = Vector{Vector{Any}}()

    for path in files
        cols = ["kappa", "lambda1", "overlap_prev"]
        data = read_columns(path, cols)
        kappa = data["kappa"]
        lambda1 = data["lambda1"]
        overlap = data["overlap_prev"]

        crossings = find_crossings(kappa, lambda1)
        min_sep = length(crossings) >= 2 ? minimum(diff(crossings)) : NaN
        clusters = cluster_crossings(crossings; delta=CROSSING_CLUSTER_DELTA)
        clustered_flag = isempty(clusters) ? 0 : 1
        cluster_spans = isempty(clusters) ? "" :
            join([@sprintf("%.4f..%.4f (n=%d)", c[1], c[2], c[3]) for c in clusters], ";")

        overlap_stats = overlap_drop_stats(kappa, overlap;
                                           abs_thresh=OVERLAP_ABS_THRESH,
                                           rel_drop=OVERLAP_REL_DROP)
        abs_kappa = [kappa[i] for i in overlap_stats.abs_idx]
        rel_kappa = [kappa[i] for i in overlap_stats.rel_idx]

        tag = parse_tag(path)
        push!(rows, Any[
            string(Dates.now()),
            split(path, '/')[end],
            tag.nu0, tag.L, tag.M,
            length(crossings),
            min_sep,
            clustered_flag,
            cluster_spans,
            overlap_stats.available ? 1 : 0,
            overlap_stats.min_overlap,
            length(abs_kappa),
            length(rel_kappa),
            format_list(abs_kappa),
            format_list(rel_kappa)
        ])
    end

    header = [
        "timestamp", "file", "nu0", "L", "M",
        "n_crossings", "min_crossing_sep",
        "clustered_crossings_flag", "cluster_spans",
        "overlap_available", "overlap_min",
        "overlap_abs_drop_count", "overlap_rel_drop_count",
        "overlap_abs_drop_kappa", "overlap_rel_drop_kappa"
    ]
    open(OUT_PATH, "w") do io
        writedlm(io, permutedims(header), ',')
        for r in rows
            writedlm(io, permutedims(r), ',')
        end
    end

    println(@sprintf("Processed %d files", length(files)))
    println("Wrote: " * OUT_PATH)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

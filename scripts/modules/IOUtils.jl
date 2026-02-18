module IOUtils

using DataFrames
using JSON3
using LibGit2

export has_col,
       get_col,
       pick_col,
       sanitize_json,
       write_docs,
       git_commit_hash

"""
    has_col(df::DataFrame, name::Symbol) -> Bool

Check whether column `name` exists in `df`. Handles Symbol vs String column names.
"""
function has_col(df::DataFrame, name::Symbol)::Bool
    s = String(name)
    return s in names(df)
end

"""
    get_col(df::DataFrame, name::Symbol) -> AbstractVector

Return column `name` from `df`, accepting Symbol or String variants.
Throws an error if column is missing.
"""
function get_col(df::DataFrame, name::Symbol)
    if name in names(df)
        return df[!, name]
    end
    s = String(name)
    if s in names(df)
        return df[!, s]
    end
    error("Column not found: $name")
end

"""
    pick_col(df::DataFrame, candidates::Vector{Symbol}) -> AbstractVector

Return the first matching column among `candidates`.
"""
function pick_col(df::DataFrame, candidates::Vector{Symbol})
    for c in candidates
        if c in names(df)
            return df[!, c]
        end
        s = String(c)
        if s in names(df)
            return df[!, s]
        end
    end
    error("None of the requested columns found: $(candidates)")
end

"""
    sanitize_json(x) -> Any

Recursively replace non-finite numbers (NaN/Inf) with `nothing` to satisfy JSON spec.
"""
function sanitize_json(x)
    if x isa Dict
        return Dict(k => sanitize_json(v) for (k, v) in x)
    elseif x isa AbstractVector
        return [sanitize_json(v) for v in x]
    elseif x isa Tuple
        return [sanitize_json(v) for v in x]
    elseif x isa Float64
        return isfinite(x) ? x : nothing
    elseif x isa Float32
        return isfinite(x) ? x : nothing
    elseif x isa Real
        return isfinite(x) ? x : nothing
    else
        return x
    end
end

"""
    write_docs(path::String, report::String)

Write a text report to disk.
"""
function write_docs(path::String, report::String)
    open(path, "w") do io
        write(io, report)
    end
end

"""
    git_commit_hash(path::String) -> String

Return current git commit hash, or "unknown" if not available.
"""
function git_commit_hash(path::String)::String
    try
        repo = LibGit2.GitRepo(path)
        return string(LibGit2.head_oid(repo))
    catch
        return "unknown"
    end
end

end # module

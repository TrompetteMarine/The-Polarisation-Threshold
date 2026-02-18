module Reporting

using Dates
using JSON3

using ..IOUtils: write_docs, sanitize_json, git_commit_hash

export write_validation_report,
       write_validation_tex,
       write_figure_captions,
       write_results_text,
       write_metadata_json

"""
    write_validation_report(path, report)

Write validation markdown report.
"""
function write_validation_report(path::String, report::String)
    write_docs(path, report)
end

"""
    write_validation_tex(path, tex)

Write LaTeX validation table snippet.
"""
function write_validation_tex(path::String, tex::String)
    write_docs(path, tex)
end

"""
    write_figure_captions(path, captions)

Write figure captions text.
"""
function write_figure_captions(path::String, captions::String)
    write_docs(path, captions)
end

"""
    write_results_text(path, text)

Write results text for manuscript.
"""
function write_results_text(path::String, text::String)
    write_docs(path, text)
end

"""
    write_metadata_json(path, metadata)

Write metadata JSON with NaN sanitization.
"""
function write_metadata_json(path::String, metadata::Dict{String,Any})
    JSON3.write(path, sanitize_json(metadata))
end

end

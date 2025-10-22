module BifurcationAnalysis

# This module is optional and only loads if BifurcationKit is available
const BIFURCATIONKIT_AVAILABLE = try
    using BifurcationKit
    true
catch
    false
end

if BIFURCATIONKIT_AVAILABLE
    include("model_interface.jl")
    include("bifurcation_core.jl")
    include("plotting_cairo.jl")
    
    using .ModelInterface
    using .BifurcationCore
    using .PlottingCairo
    
    export ModelInterface, BifurcationCore, PlottingCairo
else
    @warn "BifurcationKit not available. Bifurcation analysis features disabled."
end

end # module
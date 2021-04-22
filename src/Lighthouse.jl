module Lighthouse

using Statistics, Dates, LinearAlgebra, Random, Logging
using Base.Threads
using StatsBase: StatsBase
using TensorBoardLogger
using StatsPlots
using StatsPlots.Plots.PlotMeasures
using Printf

# Set up plotting backend for Plots.jl (GR)
function __init__()
    gr()
    GR.inline("png")
end

include("utilities.jl")
export majority

include("metrics.jl")
export confusion_matrix, accuracy, binary_statistics, cohens_kappa, calibration_curve

include("classifier.jl")
export AbstractClassifier

include("learn.jl")
export LearnLogger, learn!, upon, evaluate!, predict!

end # module

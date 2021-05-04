module Lighthouse

using Statistics, Dates, LinearAlgebra, Random, Logging
using Base.Threads
using StatsBase: StatsBase
using TensorBoardLogger
using StatsPlots
using StatsPlots.Plots.PlotMeasures
using Printf

function __init__()
    GR.inline("png")
    return nothing
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

module Lighthouse

using Statistics, Dates, LinearAlgebra, Random, Logging
using Base.Threads
using StatsBase: StatsBase
using TensorBoardLogger
using Makie
using Printf
using Legolas
using Arrow
using DataFrames

include("plotting.jl")

include("utilities.jl")
export majority

include("metrics.jl")
export confusion_matrix, accuracy, binary_statistics, cohens_kappa, calibration_curve

include("classifier.jl")
export AbstractClassifier

include("row.jl")
# TODO: export EvaluationRow ?

include("learn.jl")
export LearnLogger, learn!, upon, evaluate!, predict!

end # module

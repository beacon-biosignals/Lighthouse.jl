module Lighthouse

using Statistics, Dates, LinearAlgebra, Random, Logging
using Base.Threads
using StatsBase: StatsBase
using TensorBoardLogger
using Makie
using Printf
using Legolas
using Tables
using DataFrames

include("row.jl")
export EvaluationRow, ObservationRow

include("plotting.jl")

include("utilities.jl")
export majority

include("metrics.jl")
export confusion_matrix, accuracy, binary_statistics, cohens_kappa, calibration_curve

include("classifier.jl")
export AbstractClassifier

include("LearnLogger.jl")
export LearnLogger

include("learn.jl")
export learn!, upon, evaluate!, predict!
export log_event!, log_line_series!, log_plot!, step_logger!, log_value!, log_values!
export log_array!, log_arrays!

end # module

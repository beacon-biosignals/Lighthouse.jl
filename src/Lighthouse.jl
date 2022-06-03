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
using ArrowTypes

include("row.jl")
export EvaluationRow, ObservationRow, Curve, TradeoffMetricsRow, HardenedMetricsRow,
       LabelMetricsRow, Curve

include("plotting.jl")

include("utilities.jl")
export majority

include("metrics.jl")
export confusion_matrix, accuracy, binary_statistics, cohens_kappa, calibration_curve,
       get_tradeoff_metrics, get_tradeoff_metrics_binary_multirater, get_hardened_metrics,
       get_hardened_metrics_multirater, get_hardened_metrics_multiclass,
       get_label_metrics_multirater, get_label_metrics_multirater_multiclass,
       harden_by_threshold

include("classifier.jl")
export AbstractClassifier

include("LearnLogger.jl")
export LearnLogger

include("learn.jl")
export learn!, upon, evaluate!, predict!
export log_event!, log_line_series!, log_plot!, step_logger!, log_value!, log_values!
export log_array!, log_arrays!, log_evaluation_row!, log_tradeoff_row!, log_hardened_row!, log_labels_row!

end # module

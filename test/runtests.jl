using Test, Random, LinearAlgebra, Statistics
using Lighthouse
using Lighthouse: plot_reliability_calibration_curves, plot_prg_curves, plot_pr_curves,
                  plot_roc_curves, plot_kappas, plot_confusion_matrix, evaluation_metrics_plot,
                  plot_combined
using Base.Threads

# Set up plotting backend for Plots.jl (GR)
using Plots
gr()
GR.inline("png")

include("metrics.jl")
include("learn.jl")
include("utilities.jl")
include("logger.jl")

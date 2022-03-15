using Test, LinearAlgebra, Statistics
using StableRNGs
using Lighthouse
using Lighthouse: plot_reliability_calibration_curves, plot_pr_curves,
                  plot_roc_curves, plot_kappas, plot_confusion_matrix,
                  evaluation_metrics_plot, evaluation_metrics
using Base.Threads
using CairoMakie

# Needs to be set for figures
# returning true for showable("image/png", obj)
# which TensorBoardLogger.jl uses to determine output
CairoMakie.activate!(type="png")
plot_results = joinpath(@__DIR__, "plot_results")
# Remove any old plots
isdir(plot_results) && rm(plot_results; force=true, recursive=true)
mkdir(plot_results)

macro testplot(fig_name)
    path = joinpath(plot_results, string(fig_name, ".png"))
    return quote
        fig = $(esc(fig_name))
        @test fig isa Makie.FigureLike
        save($(path), fig)
    end
end

include("plotting.jl")
include("metrics.jl")
include("learn.jl")
include("utilities.jl")
include("logger.jl")
include("row.jl")

using Test, Random, LinearAlgebra, Statistics
using Lighthouse
using Lighthouse: plot_reliability_calibration_curves, plot_prg_curves, plot_pr_curves,
                  plot_roc_curves, plot_kappas, plot_confusion_matrix,
                  evaluation_metrics_plot
using Base.Threads
using CairoMakie

# Needs to be set for figures
# returning true for showable("image/png", obj)
# which TensorBoardLogger.jl uses to determine output
CairoMakie.activate!(type="png")
plot_results = joinpath(@__DIR__, "plot_results")
isdir(plot_results) || mkdir(plot_results)

macro testplot(fig_name)
    path = joinpath(plot_results, string(fig_name, ".png"))
    return quote
        fig = $(esc(fig_name))
        @test fig isa AbstractPlotting.FigureLike
        save($(path), fig)
    end
end

include("metrics.jl")
include("learn.jl")
include("utilities.jl")
include("logger.jl")

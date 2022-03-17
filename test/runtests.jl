using Test, LinearAlgebra, Statistics
using StableRNGs
using Lighthouse
using Lighthouse: plot_reliability_calibration_curves, plot_pr_curves,
                  plot_roc_curves, plot_kappas, plot_confusion_matrix,
                  evaluation_metrics_plot, evaluation_metrics
using Base.Threads
using CairoMakie
using Legolas, Tables

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

function test_roundtrip_evaluation(row_dict::Dict{String,S}) where {S}
    row = Lighthouse.EvaluationRow(row_dict)
    rt_row = roundtrip_row(row)

    # Make sure row roundtrips correctly
    @test issetequal(keys(row), keys(rt_row))
    for (k, v) in pairs(row)
        if ismissing(v)
            @test ismissing(rt_row[k])
        else
            @test issetequal(v, rt_row[k])
        end
    end

    # Make sure originating dictionary roundtrips correctly
    rt_dict = Lighthouse._evaluation_row_dict(rt_row)
    for (k, v) in pairs(row_dict)
        if ismissing(v)
            @test ismissing(rt_dict[k])
        else
            @test issetequal(v, rt_dict[k])
        end
    end
    return true
end

function roundtrip_row(row::Lighthouse.EvaluationRow)
    p = mktempdir() * "rt_test.arrow"
    tbl = [row]
    Legolas.write(p, tbl, Lighthouse.EVALUATION_ROW_SCHEMA)
    return Lighthouse.EvaluationRow(only(Tables.rows(Legolas.read(p))))
end

include("plotting.jl")
include("metrics.jl")
include("learn.jl")
include("utilities.jl")
include("logger.jl")
include("row.jl")

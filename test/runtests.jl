using Test, LinearAlgebra, Statistics
using StableRNGs
using Lighthouse
using Lighthouse: plot_reliability_calibration_curves, plot_pr_curves,
                  plot_roc_curves, plot_kappas, plot_confusion_matrix,
                  evaluation_metrics_plot, evaluation_metrics
using Base.Threads
using CairoMakie
using Legolas, Tables
using DataFrames

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

const EVALUATION_ROW_KEYS = string.(keys(EvaluationRow()))

function test_evaluation_metrics_roundtrip(row_dict::Dict{String,S}) where {S}
    # Make sure we're capturing all metrics keys in our Schema
    keys_not_in_schema = setdiff(keys(row_dict), EVALUATION_ROW_KEYS)
    @test isempty(keys_not_in_schema)

    # Do the roundtripping (will fail if schema types do not validate after roundtrip)
    row = EvaluationRow(row_dict)
    rt_row = roundtrip_row(row)

    # Make sure full row roundtrips correctly
    @test issetequal(keys(row), keys(rt_row))
    for (k, v) in pairs(row)
        if ismissing(v)
            @test ismissing(rt_row[k])
        else
            @test issetequal(v, rt_row[k])
        end
    end

    # Make sure originating metrics dictionary roundtrips correctly
    rt_dict = Lighthouse._evaluation_row_dict(rt_row)
    for (k, v) in pairs(row_dict)
        if ismissing(v)
            @test ismissing(rt_dict[k])
        else
            @test issetequal(v, rt_dict[k])
        end
    end
    return nothing
end

function roundtrip_row(row::EvaluationRow)
    p = mktempdir() * "rt_test.arrow"
    tbl = [row]
    Legolas.write(p, tbl, Lighthouse.EVALUATION_ROW_SCHEMA)
    return EvaluationRow(only(Tables.rows(Legolas.read(p))))
end

include("plotting.jl")
include("metrics.jl")
include("learn.jl")
include("utilities.jl")
include("logger.jl")
include("row.jl")

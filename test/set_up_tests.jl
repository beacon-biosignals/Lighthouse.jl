
using Aqua
using Arrow
using Base.Threads
using CairoMakie
using DataFrames
using Legolas, Tables
using Lighthouse
using LinearAlgebra
using StableRNGs
using Statistics
using Test

using Lighthouse: plot_reliability_calibration_curves, plot_pr_curves,
                  plot_roc_curves, plot_kappas, plot_confusion_matrix,
                  evaluation_metrics_plot, evaluation_metrics, binarize_by_threshold

# Needs to be set for figures
# returning true for showable("image/png", obj)
# which TensorBoardLogger.jl uses to determine output
CairoMakie.activate!(; type="png")
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

const EVALUATION_V1_KEYS = string.(fieldnames(EvaluationV1))

function test_evaluation_metrics_roundtrip(row_dict::Dict{String,S}) where {S}
    # Make sure all metrics keys are captured in our schema and are not thrown away
    keys_not_in_schema = setdiff(keys(row_dict), EVALUATION_V1_KEYS)
    @test isempty(keys_not_in_schema)

    # Do the roundtripping (will fail if schema types do not validate after roundtrip)
    record = EvaluationV1(row_dict)
    rt_row = roundtrip_row(record)

    # Make sure full row roundtrips correctly
    @test issetequal(keys(record), keys(rt_row))
    for (k, v) in pairs(record)
        if ismissing(v)
            @test ismissing(rt_row[k])
        else
            @test issetequal(v, rt_row[k])
        end
    end

    # Make sure originating metrics dictionary roundtrips correctly
    rt_dict = Lighthouse._evaluation_dict(rt_row)
    for (k, v) in pairs(row_dict)
        if ismissing(v)
            @test ismissing(rt_dict[k])
        else
            @test issetequal(v, rt_dict[k])
        end
    end
    return nothing
end

function roundtrip_row(row::EvaluationV1)
    io = IOBuffer()
    tbl = [row]
    Legolas.write(io, tbl, Lighthouse.EvaluationV1SchemaVersion())
    return EvaluationV1(only(Tables.rows(Legolas.read(seekstart(io)))))
end

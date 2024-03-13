# We can't rely on inference to always give us fully typed
# Vector{<: Number} so we add `{T} where T` to the the mix
# This makes the number like type a bit absurd, but is still nice for
# documentation purposes!
const NumberLike = Union{Number,Missing,Nothing,T} where {T}
const NumberVector = AbstractVector{<:NumberLike}
const NumberMatrix = AbstractMatrix{<:NumberLike}

"""
    Tuple{<:NumberVector, <: NumberVector}

Tuple of X, Y coordinates
"""
const XYVector = Tuple{<:NumberVector,<:NumberVector}

"""
    Union{XYVector, AbstractVector{<: XYVector}}

A series of XYVectors, or a single xyvector.
"""
const SeriesCurves = Union{XYVector,AbstractVector{<:XYVector}}

"""
    evaluation_metrics_plot(data::Dict; size=(1000, 1000), fontsize=12)
    evaluation_metrics_plot(row::EvaluationV1; kwargs...)

Plot all evaluation metrics generated via [`evaluation_metrics_record`](@ref) and/or
[`evaluation_metrics`](@ref) in a single image.

!!! note
    This function requires a valid Makie backend (e.g. CairoMakie) to be loaded.
"""
function evaluation_metrics_plot end

"""
    plot_confusion_matrix!(subfig::GridPosition, args...; kw...)

    plot_confusion_matrix(confusion::AbstractMatrix{<: Number},
                          class_labels::AbstractVector{String},
                          normalize_by::Union{Symbol,Nothing}=nothing;
                          size=(800,600), annotation_text_size=20)


Lighthouse plots confusion matrices, which are simple tables
showing the empirical distribution of predicted class (the rows)
versus the elected class (the columns). These can optionally be normalized:

* row-normalized (`:Row`): this means each row has been normalized to sum to 1. Thus, the row-normalized confusion matrix shows the empirical distribution of elected classes for a given predicted class. E.g. the first row of the row-normalized confusion matrix shows the empirical probabilities of the elected classes for a sample which was predicted to be in the first class.
* column-normalized (`:Column`): this means each column has been normalized to sum to 1. Thus, the column-normalized confusion matrix shows the empirical distribution of predicted classes for a given elected class. E.g. the first column of the column-normalized confusion matrix shows the empirical probabilities of the predicted classes for a sample which was elected to be in the first class.

```
fig, ax, p = plot_confusion_matrix(rand(2, 2), ["1", "2"])
fig = Figure()
ax = plot_confusion_matrix!(fig[1, 1], rand(2, 2), ["1", "2"], :Row)
ax = plot_confusion_matrix!(fig[1, 2], rand(2, 2), ["1", "2"], :Column)
```

!!! note
    This function requires a valid Makie backend (e.g. CairoMakie) to be loaded.
"""
function plot_confusion_matrix end
function plot_confusion_matrix! end

"""
    plot_reliability_calibration_curves!(fig::SubFigure, args...; kw...)

    plot_reliability_calibration_curves(per_class_reliability_calibration_curves::SeriesCurves,
                                        per_class_reliability_calibration_scores::NumberVector,
                                        class_labels::AbstractVector{String};
                                        legend=:rb, size=(800, 600))

!!! note
    This function requires a valid Makie backend (e.g. CairoMakie) to be loaded.
"""
function plot_reliability_calibration_curves end
function plot_reliability_calibration_curves! end

"""
    plot_binary_discrimination_calibration_curves!(fig::SubFigure, args...; kw...)

    plot_binary_discrimination_calibration_curves!(calibration_curve::SeriesCurves, calibration_score,
                                                   per_expert_calibration_curves::SeriesCurves,
                                                   per_expert_calibration_scores, optimal_threshold,
                                                   discrimination_class::AbstractString;
                                                   marker=:rect, markersize=5, linewidth=2)

!!! note
    This function requires a valid Makie backend (e.g. CairoMakie) to be loaded.
"""
function plot_binary_discrimination_calibration_curves end
function plot_binary_discrimination_calibration_curves! end

"""
    plot_pr_curves!(subfig::GridPosition, args...; kw...)

    plot_pr_curves(per_class_pr_curves::SeriesCurves,
                class_labels::AbstractVector{<: String};
                size=(800, 600),
                legend=:lt, title="PR curves",
                xlabel="True positive rate", ylabel="Precision",
                linewidth=2, scatter=NamedTuple(), color=:darktest)

- `scatter::Union{Nothing, NamedTuple}`: can be set to a named tuples of attributes that are forwarded to the scatter call (e.g. markersize). If nothing, no scatter is added.


!!! note
    This function requires a valid Makie backend (e.g. CairoMakie) to be loaded.
"""
function plot_pr_curves end
function plot_pr_curves! end

"""
    plot_roc_curves!(subfig::GridPosition, args...; kw...)

    plot_roc_curves(per_class_roc_curves::SeriesCurves,
                    per_class_roc_aucs::NumberVector,
                    class_labels::AbstractVector{<: String};
                    size=(800, 600),
                    legend=:lt,
                    title="ROC curves",
                    xlabel="False positive rate",
                    ylabel="True positive rate",
                    linewidth=2, scatter=NamedTuple(), color=:darktest)

- `scatter::Union{Nothing, NamedTuple}`: can be set to a named tuples of attributes that are forwarded to the scatter call (e.g. markersize). If nothing, no scatter is added.

!!! note
    This function requires a valid Makie backend (e.g. CairoMakie) to be loaded.
"""
function plot_roc_curves end
function plot_roc_curves! end

"""
    plot_kappas!(subfig::GridPosition, args...; kw...)

    plot_kappas(per_class_kappas::NumberVector,
                class_labels::AbstractVector{String},
                per_class_IRA_kappas=nothing;
                size=(800, 600),
                annotation_text_size=20)

!!! note
    This function requires a valid Makie backend (e.g. CairoMakie) to be loaded.
"""
function plot_kappas end
function plot_kappas! end

#####
##### Deprecation support
#####

"""
    evaluation_metrics_plot(predicted_hard_labels::AbstractVector,
                            predicted_soft_labels::AbstractMatrix,
                            elected_hard_labels::AbstractVector,
                            classes,
                            thresholds=0.0:0.01:1.0;
                            votes::Union{Nothing,AbstractMatrix}=nothing,
                            strata::Union{Nothing,AbstractVector{Set{T}} where T}=nothing,
                            optimal_threshold_class::Union{Nothing,Integer}=nothing)

Return a plot and dictionary containing a battery of classifier performance
metrics that each compare `predicted_soft_labels` and/or `predicted_hard_labels`
agaist `elected_hard_labels`.

See [`evaluation_metrics`](@ref) for a description of the arguments.

This method is deprecated in favor of calling `evaluation_metrics`
and [`evaluation_metrics_plot`](@ref) separately.

!!! note
    This function requires a valid Makie backend (e.g. CairoMakie) to be loaded.
"""
function evaluation_metrics_plot end

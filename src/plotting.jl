if !isdefined(Makie, :FigurePosition)
    const FigurePosition = Makie.GridPosition
    get_parent(x::Makie.GridPosition) = x.layout.parent
else
    using Makie: FigurePosition
    get_parent(x::FigurePosition) = x.fig
end

# We can't rely on inference to always give us fully typed
# Vector{<: Number} so we add `{T} where T` to the the mix
# This makes the number like type a bit absurd, but is still nice for
# documentation purposes!
const NumberLike = Union{Number,Missing,Nothing,T} where {T}
const NumberVector = AbstractVector{<:NumberLike}
const NumberMatrix = AbstractMatrix{<:NumberLike}

#####
##### Helpers for theming and color generation...May want to move them to Colors.jl / Makie.jl
#####
using Makie.Colors: LCHab, distinguishable_colors, RGB, Colorant

function get_theme(scene, key1::Symbol, key2::Symbol; defaults...)
    return get_theme(Makie.get_scene(scene), key1, key2; defaults...)
end

function get_theme(fig::FigurePosition, key1::Symbol, key2::Symbol; defaults...)
    return get_theme(get_parent(fig), key1, key2; defaults...)
end

# This function helps us to get the theme from a scene, that we can apply to our plotting functions
function get_theme(scene::Scene, key1::Symbol, key2::Symbol; defaults...)
    scene_theme = theme(scene)
    sub_theme = get(scene_theme, key1, Theme())
    # The priority is key1.key2 > defaults > key2
    # this way defaults are overwritten by theme options specifically set for our theme.
    # Consider Kappas.Axis for key1/2, what we want is, that if there are defaults in Kappas.Axis
    # they should overwrite our generic lighthouse defaults. But anything not specified in Kappas.Axis/defaults,
    # should fall back to scene_theme.Axis
    return merge(get(sub_theme, key2, Theme()), Theme(; defaults...),
                 get(scene_theme, key2, Theme()))
end

function high_contrast(background_color::Colorant, target_color::Colorant;
                       # chose from whole lightness spectrum
                       lchoices=range(0; stop=100, length=15))
    target = LCHab(target_color)
    color = distinguishable_colors(1, [RGB(background_color)]; dropseed=true,
                                   lchoices=lchoices,
                                   cchoices=[target.c], hchoices=[target.h])
    return RGBAf(color[1], Makie.Colors.alpha(target_color))
end

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

function series_plot!(subfig::FigurePosition, per_class_pr_curves::SeriesCurves,
                      class_labels::Union{Nothing,AbstractVector{String}}; legend=:lt,
                      title="No title",
                      xlabel="x label", ylabel="y label", solid_color=nothing,
                      color=nothing,
                      linewidth=nothing, scatter=NamedTuple())
    axis_theme = get_theme(subfig, :SeriesPlot, :Axis; title=title, titlealign=:left,
                           xlabel=xlabel,
                           ylabel=ylabel, aspect=AxisAspect(1),
                           xticks=0:0.2:1, yticks=0.2:0.2:1)

    ax = Axis(subfig; axis_theme...)
    # Not the most elegant, but this way we can let the theming to the Series theme, or
    # pass it through explicitely
    series_theme = get_theme(subfig, :SeriesPlot, :Series; scatter...)
    isnothing(solid_color) || (series_theme[:solid_color] = solid_color)
    isnothing(color) || (series_theme[:color] = color)
    isnothing(linewidth) || (series_theme[:linewidth] = linewidth)
    series_theme = merge(series_theme, Attributes(; scatter...))
    hidedecorations!(ax; label=false, ticklabels=false, grid=false)
    limits!(ax, 0, 1, 0, 1)
    Makie.series!(ax, per_class_pr_curves; labels=class_labels, series_theme...)
    if !isnothing(legend)
        axislegend(ax; position=legend)
    end
    return ax
end

function plot_pr_curves!(subfig::FigurePosition, per_class_pr_curves::SeriesCurves,
                         class_labels::Union{Nothing,AbstractVector{String}}; legend=:lt,
                         title="PR curves",
                         xlabel="True positive rate", ylabel="Precision",
                         scatter=NamedTuple(),
                         solid_color=nothing)
    return series_plot!(subfig, per_class_pr_curves, class_labels; legend=legend,
                        title=title, xlabel=xlabel,
                        ylabel=ylabel, scatter=scatter, solid_color=solid_color)
end

function plot_roc_curves!(subfig::FigurePosition, per_class_roc_curves::SeriesCurves,
                          per_class_roc_aucs::NumberVector,
                          class_labels::AbstractVector{<:String};
                          legend=:rb, title="ROC curves", xlabel="False positive rate",
                          ylabel="True positive rate")
    auc_labels = [@sprintf("%s (AUC: %.3f)", class, per_class_roc_aucs[i])
                  for (i, class) in enumerate(class_labels)]

    return series_plot!(subfig, per_class_roc_curves, auc_labels; legend=legend,
                        title=title, xlabel=xlabel,
                        ylabel=ylabel)
end

function plot_reliability_calibration_curves!(subfig::FigurePosition,
                                              per_class_reliability_calibration_curves::SeriesCurves,
                                              per_class_reliability_calibration_scores::NumberVector,
                                              class_labels::AbstractVector{String};
                                              legend=:rb)
    calibration_score_labels = map(enumerate(class_labels)) do (i, class)
        @sprintf("%s (MSE: %.3f)", class, per_class_reliability_calibration_scores[i])
    end

    scatter_theme = get_theme(subfig, :ReliabilityCalibrationCurves, :Scatter;
                              marker=Circle,
                              markersize=5, strokewidth=0)
    ideal_theme = get_theme(subfig, :ReliabilityCalibrationCurves, :Ideal;
                            color=(:black, 0.5),
                            linestyle=:dash, linewidth=2)

    ax = series_plot!(subfig, per_class_reliability_calibration_curves,
                      calibration_score_labels;
                      legend=legend, title="Prediction reliability calibration",
                      xlabel="Predicted probability bin", ylabel="Fraction of positives",
                      scatter=scatter_theme)
    #TODO: mean predicted value histogram underneath?? Maybe important...
    # https://scikit-learn.org/stable/modules/calibration.html
    linesegments!(ax, [0, 1], [0, 1]; ideal_theme...)
    return ax
end

function set_from_kw!(theme, key, kw, default)
    if haskey(kw, key)
        theme[key] = getproperty(kw, key)
    elseif !haskey(theme, key)
        theme[key] = default
    end
    return
end

function plot_binary_discrimination_calibration_curves!(subfig::FigurePosition,
                                                        calibration_curve::SeriesCurves,
                                                        calibration_score,
                                                        per_expert_calibration_curves::SeriesCurves,
                                                        per_expert_calibration_scores,
                                                        optimal_threshold,
                                                        discrimination_class::AbstractString;
                                                        kw...)
    kw = values(kw)
    scatter_theme = get_theme(subfig, :BinaryDiscriminationCalibrationCurves, :Scatter;
                              strokewidth=0)
    # Hayaah, this theme merging is getting out of hand
    # but we want kw > BinaryDiscriminationCalibrationCurves > Scatter, so we need to somehow set things
    # after the theme merging above, especially, since we also pass those to series!,
    # which then again tries to merge the kw args with the theme.
    set_from_kw!(scatter_theme, :makersize, kw, 5)
    set_from_kw!(scatter_theme, :marker, kw, :rect)

    per_expert = get_theme(subfig, :BinaryDiscriminationCalibrationCurves, :PerExpert;
                           solid_color=:darkgrey,
                           color=nothing)
    set_from_kw!(per_expert, :linewidth, kw, 2)
    ax = series_plot!(subfig, per_expert_calibration_curves, nothing; legend=nothing,
                      title="Detection calibration", xlabel="Expert agreement rate",
                      ylabel="Predicted positive probability", scatter=scatter_theme,
                      per_expert...)

    calibration = get_theme(subfig, :BinaryDiscriminationCalibrationCurves,
                            :CalibrationCurve;
                            solid_color=:navyblue, markerstrokewidth=0)

    set_from_kw!(calibration, :markersize, kw, 5)
    set_from_kw!(calibration, :marker, kw, :rect)
    set_from_kw!(calibration, :linewidth, kw, 2)

    Makie.series!(ax, calibration_curve; calibration...)

    ideal_theme = get_theme(subfig, :BinaryDiscriminationCalibrationCurves, :Ideal;
                            color=(:black, 0.5),
                            linestyle=:dash)
    set_from_kw!(ideal_theme, :linewidth, kw, 2)
    linesegments!(ax, [0, 1], [0, 1]; label="Ideal", ideal_theme...)
    #TODO: expert agreement histogram underneath?? Maybe important...
    # https://scikit-learn.org/stable/modules/calibration.html
    return ax
end

function plot_confusion_matrix!(subfig::FigurePosition, confusion::NumberMatrix,
                                class_labels::AbstractVector{String}, normalize_by::Symbol;
                                annotation_text_size=20, colormap=:Blues)
    normdim = get((Row=2, Column=1), normalize_by) do
        return error("normalize_by must be either :Row or :Column, found: $(normalize_by)")
    end

    nclasses = length(class_labels)
    if size(confusion) != (nclasses, nclasses)
        error("Labels must match size of square confusion matrix. Found $(nclasses) labels for an $(size(confusion)) matrix")
    end
    confusion = round.(confusion ./ sum(confusion; dims=normdim); digits=3)
    class_indices = 1:nclasses
    text_theme = get_theme(subfig, :ConfusionMatrix, :Text; textsize=annotation_text_size)
    heatmap_theme = get_theme(subfig, :ConfusionMatrix, :Heatmap; nan_color=(:black, 0.0))
    axis_theme = get_theme(subfig, :ConfusionMatrix, :Axis; xticklabelrotation=pi / 4,
                           titlealign=:left,
                           title="$(string(normalize_by))-Normalized Confusion",
                           xlabel="Elected Class", ylabel="Predicted Class",
                           xticks=(class_indices, class_labels),
                           yticks=(class_indices, class_labels),
                           aspect=AxisAspect(1))

    ax = Axis(subfig; axis_theme...)

    hidedecorations!(ax; label=false, ticklabels=false, grid=false)
    ylims!(ax, nclasses + 0.5, 0.5)
    tightlimits!(ax)
    plot_bg_color = to_color(ax.backgroundcolor[])
    crange = (0.0, 1.0)
    nan_color = to_color(heatmap_theme.nan_color[])
    cmap = to_colormap(to_value(pop!(heatmap_theme, :colormap, colormap)))
    heatmap!(ax, confusion'; colorrange=crange, colormap=cmap, nan_color=nan_color,
             heatmap_theme...)
    text_color = to_color(to_value(pop!(text_theme, :color, :black)))
    function label_color(i, j)
        c = confusion[i, j]
        bg_color = if isnan(c) || ismissing(c)
            Makie.Colors.alpha(nan_color) <= 0.0 ? plot_bg_color : nan_color
        else
            Makie.interpolated_getindex(cmap, c, crange)
        end
        return high_contrast(bg_color, text_color)
    end
    annos = vec([(string(confusion[i, j]), Point2f(j, i))
                 for i in class_indices, j in class_indices])
    colors = vec([label_color(i, j) for i in class_indices, j in class_indices])
    text!(ax, annos; align=(:center, :center), color=colors, textsize=annotation_text_size,
          text_theme...)
    return ax
end

function text_attributes(values, groups, bar_colors, bg_color, text_color)
    aligns = NTuple{2,Symbol}[]
    offsets = NTuple{2,Int}[]
    text_colors = RGBAf[]
    for (i, k) in enumerate(values)
        group = groups isa AbstractVector ? groups[i] : groups
        bar_color = bar_colors[group]
        # Plot text inside bar
        if k > 0.85
            push!(aligns, (:right, :center))
            push!(offsets, (-10, 0))
            push!(text_colors, high_contrast(bar_color, text_color))
        else
            # plot text next to bar
            push!(aligns, (:left, :center))
            push!(offsets, (10, 0))
            push!(text_colors, high_contrast(bg_color, text_color))
        end
    end
    return aligns, offsets, text_colors
end

function plot_kappas!(subfig::FigurePosition, per_class_kappas::NumberVector,
                      class_labels::AbstractVector{String}, per_class_IRA_kappas=nothing;
                      color=[:lightgrey, :lightblue], annotation_text_size=20)
    nclasses = length(class_labels)
    axis_theme = get_theme(subfig, :Kappas, :Axis; aspect=AxisAspect(1), titlealign=:left,
                           xlabel="Cohen's kappa", xticks=[0, 1], yreversed=true,
                           yticks=(1:nclasses, class_labels))

    text_theme = get_theme(subfig, :Kappas, :Text; textsize=annotation_text_size)
    text_color = to_color(to_value(pop!(text_theme, :color, to_color(:black))))
    bars_theme = get_theme(subfig, :Kappas, :BarPlot; color=color)
    bar_colors = to_color.(bars_theme.color[])

    ax = Axis(subfig[1, 1]; axis_theme...)
    bg_color = to_color(ax.backgroundcolor[])
    xlims!(ax, 0, 1)
    if !has_value(per_class_IRA_kappas)
        ax.title = "Algorithm-expert agreement"
        annotations = map(enumerate(per_class_kappas)) do (i, k)
            return (string(round(k; digits=3)), Point2f(max(0, k), i))
        end
        aligns, offsets, text_colors = text_attributes(per_class_kappas, 2, bar_colors,
                                                       bg_color, text_color)
        barplot!(ax, per_class_kappas; direction=:x, color=bar_colors[2])
        text!(ax, annotations; align=aligns, offset=offsets, color=text_colors,
              text_theme...)
    else
        ax.title = "Inter-rater reliability"
        values = vcat(per_class_kappas, per_class_IRA_kappas)
        groups = vcat(fill(2, nclasses), fill(1, nclasses))
        xvals = vcat(1:nclasses, 1:nclasses)
        cmap = bar_colors
        bars = barplot!(ax, xvals, max.(0, values); dodge=groups, color=groups,
                        direction=:x,
                        colormap=cmap)
        # This is a bit hacky, but for now the easiest way to figure out the exact, dodged positions
        rectangles = bars.plots[][1][]
        dodged_y = last.(minimum.(rectangles)) .+ (last.(widths.(rectangles)) ./ 2)
        textpos = Point2f.(max.(0, values), dodged_y)

        labels = string.(round.(values; digits=3))
        aligns, offsets, text_colors = text_attributes(values, groups, bar_colors, bg_color,
                                                       text_color)
        text!(ax, labels; position=textpos, align=aligns, offset=offsets, color=text_colors,
              text_theme...)
        labels = ["Expert-vs-expert IRA", "Algorithm-vs-expert"]
        entries = map(c -> PolyElement(; color=c, strokewidth=0, strokecolor=:white), cmap)
        legend = Legend(subfig[1, 1, Bottom()], entries, labels; tellwidth=false,
                        tellheight=true,
                        labelsize=12, padding=(0, 0, 0, 0), framevisible=false,
                        patchsize=(10, 10),
                        patchlabelgap=6, labeljustification=:left)
        legend.margin = (0, 0, 0, 60)
    end
    return ax
end

"""
    evaluation_metrics_plot(data::Dict; resolution=(1000, 1000), textsize=12)
    evaluation_metrics_plot(row::EvaluationRow; kwargs...)

Plot all evaluation metrics generated via [`evaluation_metrics_row`](@ref) and/or
[`evaluation_metrics`](@ref) in a single image.
"""
function evaluation_metrics_plot(data::Dict; kwargs...)
    return evaluation_metrics_plot(EvaluationRow(data); kwargs...)
end

function evaluation_metrics_plot(row::EvaluationRow; resolution=(1000, 1000),
                                 textsize=12)
    fig = Figure(; resolution=resolution, Axis=(titlesize=17,))

    # Confusion
    plot_confusion_matrix!(fig[1, 1], row.confusion_matrix, row.class_labels,
                           :Column;
                           annotation_text_size=textsize)
    plot_confusion_matrix!(fig[1, 2], row.confusion_matrix, row.class_labels, :Row;
                           annotation_text_size=textsize)
    # Kappas
    IRA_kappa_data = nothing
    multiclass = length(row.class_labels) > 2
    labels = multiclass ? vcat("Multiclass", row.class_labels) : row.class_labels
    kappa_data = multiclass ? vcat(row.multiclass_kappa, row.per_class_kappas) :
                 row.per_class_kappas

    if issubset([:multiclass_IRA_kappas, :per_class_IRA_kappas], keys(row))
        IRA_kappa_data = multiclass ?
                         vcat(row.multiclass_IRA_kappas, row.per_class_IRA_kappas) :
                         row.per_class_IRA_kappas
    end

    plot_kappas!(fig[1, 3], kappa_data, labels, IRA_kappa_data;
                 annotation_text_size=textsize)

    # Curves
    ax = plot_roc_curves!(fig[2, 1], row.per_class_roc_curves,
                          row.per_class_roc_aucs,
                          row.class_labels; legend=nothing)

    plot_pr_curves!(fig[2, 2], row.per_class_pr_curves, row.class_labels;
                    legend=nothing)

    plot_reliability_calibration_curves!(fig[3, 1],
                                         row.per_class_reliability_calibration_curves,
                                         row.per_class_reliability_calibration_scores,
                                         row.class_labels; legend=nothing)

    legend_pos = 2:3
    if has_value(row.discrimination_calibration_curve)
        legend_pos = 3
        plot_binary_discrimination_calibration_curves!(fig[3, 2],
                                                       row.discrimination_calibration_curve,
                                                       row.discrimination_calibration_score,
                                                       row.per_expert_discrimination_calibration_curves,
                                                       row.per_expert_discrimination_calibration_scores,
                                                       row.optimal_threshold,
                                                       row.class_labels[row.optimal_threshold_class])
    end
    legend_plots = filter(Makie.MakieLayout.get_plots(ax)) do plot
        return haskey(plot, :label)
    end
    elements = map(legend_plots) do elem
        return [PolyElement(; color=elem.color, strokecolor=:transparent)]
    end

    function label_str(i)
        auc = round(row.per_class_roc_aucs[i]; digits=2)
        mse = round(row.per_class_reliability_calibration_scores[i]; digits=2)
        return ["""ROC AUC  $auc
                   Cal. MSE    $mse
                   """]
    end
    classes = row.class_labels
    nclasses = length(classes)
    class_labels = label_str.(1:nclasses)
    Legend(fig[3, legend_pos], elements, class_labels, classes; nbanks=2, tellwidth=false,
           tellheight=false,
           labelsize=11, titlesize=14, titlegap=5, groupgap=6, labelhalign=:left,
           labelvalign=:center)
    colgap!(fig.layout, 2)
    rowgap!(fig.layout, 4)
    return fig
end

# Helper to more easily define the non mutating versions
function axisplot(func, args; resolution=(800, 600), plot_kw...)
    fig = Figure(; resolution=resolution)
    ax = func(fig[1, 1], args...; plot_kw...)
    # ax.plots[1] is not really that great, but there isn't a FigureAxis object right now
    # this will need to wait for when we figure out a better recipe integration
    return Makie.FigureAxisPlot(fig, ax, ax.scene.plots[1])
end

"""
    plot_confusion_matrix!(subfig::FigurePosition, args...; kw...)

    plot_confusion_matrix(confusion::AbstractMatrix{<: Number}, class_labels::AbstractVector{String}, normalize_by::Symbol;
                          resolution=(800,600),
                          annotation_text_size=20)


Lighthouse plots confusion matrices, which are simple tables
showing the empirical distribution of predicted class (the rows)
versus the elected class (the columns). These come in two variants:

* row-normalized: this means each row has been normalized to sum to 1. Thus, the row-normalized confusion matrix shows the empirical distribution of elected classes for a given predicted class. E.g. the first row of the row-normalized confusion matrix shows the empirical probabilities of the elected classes for a sample which was predicted to be in the first class.
* column-normalized: this means each column has been normalized to sum to 1. Thus, the column-normalized confusion matrix shows the empirical distribution of predicted classes for a given elected class. E.g. the first column of the column-normalized confusion matrix shows the empirical probabilities of the predicted classes for a sample which was elected to be in the first class.

```
fig, ax, p = plot_confusion_matrix(rand(2, 2), ["1", "2"], :Row)
fig = Figure()
ax = plot_confusion_matrix!(fig[1, 1], rand(2, 2), ["1", "2"], :Column)
```
"""
plot_confusion_matrix(args...; kw...) = axisplot(plot_confusion_matrix!, args; kw...)

"""
    plot_reliability_calibration_curves!(fig::SubFigure, args...; kw...)

    plot_reliability_calibration_curves(per_class_reliability_calibration_curves::SeriesCurves,
                                        per_class_reliability_calibration_scores::NumberVector,
                                        class_labels::AbstractVector{String};
                                        legend=:rb, resolution=(800, 600))

"""
function plot_reliability_calibration_curves(args...; kw...)
    return axisplot(plot_reliability_calibration_curves!, args; kw...)
end

"""
    plot_binary_discrimination_calibration_curves!(fig::SubFigure, args...; kw...)

    plot_binary_discrimination_calibration_curves!(calibration_curve::SeriesCurves, calibration_score,
                                                   per_expert_calibration_curves::SeriesCurves,
                                                   per_expert_calibration_scores, optimal_threshold,
                                                   discrimination_class::AbstractString;
                                                   marker=:rect, markersize=5, linewidth=2)

"""
function plot_binary_discrimination_calibration_curves(args...; kw...)
    return axisplot(plot_binary_discrimination_calibration_curves!, args; kw...)
end

"""
    plot_pr_curves!(subfig::FigurePosition, args...; kw...)

    plot_pr_curves(per_class_pr_curves::SeriesCurves,
                class_labels::AbstractVector{<: String};
                resolution=(800, 600),
                legend=:lt, title="PR curves",
                xlabel="True positive rate", ylabel="Precision",
                linewidth=2, scatter=NamedTuple(), color=:darktest)

- `scatter::Union{Nothing, NamedTuple}`: can be set to a named tuples of attributes that are forwarded to the scatter call (e.g. markersize). If nothing, no scatter is added.

"""
plot_pr_curves(args...; kw...) = axisplot(plot_pr_curves!, args; kw...)

"""
    plot_roc_curves!(subfig::FigurePosition, args...; kw...)

    plot_roc_curves(per_class_roc_curves::SeriesCurves,
                    per_class_roc_aucs::NumberVector,
                    class_labels::AbstractVector{<: String};
                    resolution=(800, 600),
                    legend=:lt,
                    title="ROC curves",
                    xlabel="False positive rate",
                    ylabel="True positive rate",
                    linewidth=2, scatter=NamedTuple(), color=:darktest)

- `scatter::Union{Nothing, NamedTuple}`: can be set to a named tuples of attributes that are forwarded to the scatter call (e.g. markersize). If nothing, no scatter is added.

"""
plot_roc_curves(args...; kw...) = axisplot(plot_roc_curves!, args; kw...)

"""
    plot_kappas!(subfig::FigurePosition, args...; kw...)

    plot_kappas(per_class_kappas::NumberVector,
                class_labels::AbstractVector{String},
                per_class_IRA_kappas=nothing;
                resolution=(800, 600),
                annotation_text_size=20)
"""
plot_kappas(args...; kw...) = axisplot(plot_kappas!, args; kw...)

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
"""
function evaluation_metrics_plot(predicted_hard_labels::AbstractVector,
                                 predicted_soft_labels::AbstractMatrix,
                                 elected_hard_labels::AbstractVector, classes, thresholds;
                                 votes::Union{Nothing,AbstractMatrix}=nothing,
                                 strata::Union{Nothing,AbstractVector{Set{T}} where T}=nothing,
                                 optimal_threshold_class::Union{Nothing,Integer}=nothing)
    Base.depwarn("""
    ```
    evaluation_metrics_plot(predicted_hard_labels::AbstractVector,
                            predicted_soft_labels::AbstractMatrix,
                            elected_hard_labels::AbstractVector, classes, thresholds;
                            votes::Union{Nothing,AbstractMatrix}=nothing,
                            strata::Union{Nothing,AbstractVector{Set{T}} where T}=nothing,
                            optimal_threshold_class::Union{Nothing,Integer}=nothing)
    ```
    has been deprecated in favor of
    ```
    plot_dict = evaluation_metrics(predicted_hard_labels, predicted_soft_labels,
                                   elected_hard_labels, classes, thresholds;
                                   votes, strata, optimal_threshold_class)
    (evaluation_metrics_plot(plot_dict), plot_dict)
    ```
    """, :evaluation_metrics_plot)
    plot_dict = evaluation_metrics(predicted_hard_labels, predicted_soft_labels,
                                   elected_hard_labels, classes, thresholds; votes, strata,
                                   optimal_threshold_class)
    return evaluation_metrics_plot(plot_dict), plot_dict
end

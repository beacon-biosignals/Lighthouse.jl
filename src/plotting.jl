replace_missings(x) = replace(x, missing => NaN)

function series!(ax::Axis, curves::Tuple{<:AbstractVector, <: AbstractVector}; kw...)
    series!(ax, [curves]; kw...)
end

function series!(ax::Axis, curves; labels=nothing, linewidth=2, scatter=nothing, color=nothing)
    colors = to_colormap(:lighttest, 8)
    for (i, (x, y)) in enumerate(curves)
        x = replace_missings(x)
        y = replace_missings(y)
        label = isnothing(labels) ? "" : labels[i]
        series_color = isnothing(color) ? colors[i] : color
        if !isnothing(scatter)
            scatterlines!(ax, x, y; linewidth=linewidth, color=series_color, markercolor=series_color,
                          label=label, scatter...)
        else
            lines!(ax, x, y; linewidth=linewidth, color=series_color, label=label)
        end
    end
end

function plot_pr_curves!(subfig, per_class_pr_curves, class_labels; legend=:lb, titel="PR curves",
                         xlabel="True positive rate", ylabel="Precision", linewidth=2, scatter=nothing,
                         color=nothing)
    ax = Axis(subfig; title=titel, xlabel=xlabel, ylabel=ylabel, xticks=0:0.2:1, yticks=0:0.2:1)
    xlims!(ax, 0, 1)
    ylims!(ax, 0, 1)
    series!(ax, per_class_pr_curves; labels=class_labels, linewidth=linewidth, scatter=scatter, color=color)
    if !isnothing(legend)
        axislegend(ax; position=legend)
    end
    return ax
end

function plot_prg_curves!(subfig, per_class_prg_curves, per_class_prg_aucs, class_labels; legend=:lb)
    auc_labels = [@sprintf("%s (AUC F1: %.3f)", class, per_class_prg_aucs[i])
                  for (i, class) in enumerate(class_labels)]
    return plot_pr_curves!(subfig, per_class_prg_curves, auc_labels; legend=legend, titel="PR-Gain curves",
                           xlabel="True positive rate gain", ylabel="Precision gain")
end

function plot_roc_curves!(subfig, per_class_roc_curves, per_class_roc_aucs, class_labels; legend=:rb)
    auc_labels = [@sprintf("%s (AUC: %.3f)", class, per_class_roc_aucs[i])
                  for (i, class) in enumerate(class_labels)]

    return plot_pr_curves!(subfig, per_class_roc_curves, auc_labels; legend=legend, titel="ROC curves",
                           xlabel="False positive rate", ylabel="True positive rate")
end

function plot_reliability_calibration_curves!(subfig, per_class_reliability_calibration_curves,
                                              per_class_reliability_calibration_scores, class_labels;
                                              legend=:rb)
    calibration_score_labels = map(enumerate(class_labels)) do (i, class)
        @sprintf("%s (MSE: %.3f)", class, per_class_reliability_calibration_scores[i])
    end

    ax = plot_pr_curves!(subfig, per_class_reliability_calibration_curves, calibration_score_labels;
                         legend=legend, titel="Prediction reliability calibration",
                         xlabel="Predicted probability bin", ylabel="Fraction of positives",
                         scatter=(markershape=Circle, markersize=5, markerstroke=0))
    #TODO: mean predicted value histogram underneath?? Maybe important...
    # https://scikit-learn.org/stable/modules/calibration.html
    linesegments!(ax, [0, 1], [0, 1]; color=(:black, 0.5), linewidth=2, linestyle=:dash, label="Ideal")
    return ax
end

function plot_binary_discrimination_calibration_curves!(subfig, calibration_curve, calibration_score,
                                                        per_expert_calibration_curves,
                                                        per_expert_calibration_scores, optimal_threshold,
                                                        discrimination_class::AbstractString;
                                                        markershape=Rect, markersize=5)
    ax = plot_pr_curves!(subfig, per_expert_calibration_curves, nothing; legend=nothing,
                         titel="Detection calibration", xlabel="Expert agreement rate",
                         ylabel="Predicted positive probability", color=:darkgrey,
                         scatter=(markershape=markershape, markersize=markersize))

    scatter = (markershape=:circle, markersize=markersize, markerstrokewidth=0, color=:navyblue)
    series!(ax, calibration_curve; scatter=scatter, color=:navyblue, linewidth=1)
    linesegments!(ax, [0, 1], [0, 1]; color=(:black, 0.5), linewidth=2, linestyle=:dash, label="Ideal")
    #TODO: expert agreement histogram underneath?? Maybe important...
    # https://scikit-learn.org/stable/modules/calibration.html
    return ax
end

function plot_confusion_matrix!(subfig, confusion::AbstractMatrix, class_labels, normalize_by::Symbol;
                                annotation_text_size=20)
    normdim = get((Row=2, Column=1), normalize_by) do
        return error("normalize_by must be either :Row or :Column, found: $(normalize_by)")
    end

    nclasses = length(class_labels)
    if size(confusion) != (nclasses, nclasses)
        error("Labels must match size of square confusion matrix. Found $(nclasses) labels for an $(size(confusion)) matrix")
    end
    confusion = round.(confusion ./ sum(confusion; dims=normdim); digits=3)
    class_indices = 1:nclasses

    ax = Axis(subfig; title="$(string(normalize_by))-Normalized Confusion", xlabel="Elected Class",
              ylabel="Predicted Class", clims=(0, maximum(confusion)), xticks=(class_indices, class_labels),
              yticks=(class_indices, class_labels), xticklabelrotation=pi / 4)

    ylims!(ax, nclasses, 0)

    tightlimits!(ax)
    # Really unfortunate, that heatmap is not correctly aligned
    aligned = range(0.5; stop=nclasses + 0.5, length=nclasses)
    heatmap!(ax, aligned, aligned, confusion'; colormap=:Blues, colornorm=(0, maximum(confusion)))

    annos = vec([(string(confusion[i, j]), Point2f0(j, i)) for i in class_indices, j in class_indices])
    text!(ax, annos; align=(:center, :center), color=:black, textsize=annotation_text_size)
    return ax
end

function plot_kappas!(subfig, per_class_kappas, class_labels, per_class_IRA_kappas=nothing;
                      annotation_text_size=20)
    # Note: both the data and the labels need to be reversed, so that it plots
    # with the first class at the top of plot.

    nclasses = length(class_labels)
    ax = Axis(subfig[1, 1]; title="Algorithm-expert agreement", xlabel="Cohen's kappa",
              yticks=(1:nclasses, class_labels))

    hidedecorations!(ax; label=false, ticklabels=false, ticks=false)
    ylims!(ax, nclasses + 1, 0)
    xlims!(ax, 0, 1)
    if isnothing(per_class_IRA_kappas)
        annotations = map(enumerate(per_class_kappas)) do (i, k)
            return (string(round(k; digits=3)), Point2f0(max(0, k), i))
        end
        barplot!(ax, per_class_kappas; direction=:x, color=:lightblue)
        text!(ax, annotations; align=(:left, :center), offset=(10, 0), textsize=annotation_text_size)
    else
        values = vcat(per_class_kappas, per_class_IRA_kappas)
        groups = vcat(fill(1, nclasses), fill(2, nclasses))
        xvals = vcat(1:nclasses, 1:nclasses)
        cmap = to_color.([:lightblue :lightgrey])
        bars = barplot!(ax, xvals, max.(0, values); dodge=groups, color=groups, direction=:x, colormap=cmap)
        # This is a bit hacky, but for now the easiest way to figure out the exact, dodged positions
        rectangles = bars.plots[][1][]
        dodged_y = last.(minimum.(rectangles)) .+ (last.(widths.(rectangles)) ./ 2)
        textpos = Point2f0.(max.(0, values), dodged_y)
        labels = string.(round.(values; digits=3))
        text!(ax, labels; position=textpos, align=(:left, :center), offset=(10, 0),
              textsize=annotation_text_size)
        labels = ["Expert-vs-expert IRA", "Algorithm-vs-expert"]
        entries = map(c -> PolyElement(; color=c, strokewidth=0, strokecolor=:white), cmap)
        legend = Legend(subfig[1, 1, Bottom()], entries, labels; tellwidth=false, tellheight=true,
                        labelsize=12, padding=(0, 0, 0, 0), framevisible=false, patchsize=(10, 10),
                        patchlabelgap=2)
        legend.margin = (0, 0, 0, 60)
    end
    return ax
end

function evaluation_metrics_plot(plot_data::Dict; resolution=(1000, 1000), textsize=12)
    fig = Figure(; resolution=resolution)

    # Confusion
    confusion_row = plot_confusion_matrix!(fig[1, 1], plot_data["confusion_matrix"],
                                           plot_data["class_labels"], :Row; annotation_text_size=textsize)
    confusion_col = plot_confusion_matrix!(fig[1, 2], plot_data["confusion_matrix"],
                                           plot_data["class_labels"], :Column; annotation_text_size=textsize)
    # Kappas
    IRA_kappa_data = nothing
    multiclass = length(plot_data["class_labels"]) > 2
    labels = multiclass ? hcat("Multiclass", plot_data["class_labels"]) : plot_data["class_labels"]
    kappa_data = multiclass ? vcat(plot_data["multiclass_kappa"], plot_data["per_class_kappas"]) :
                 plot_data["per_class_kappas"]

    if issubset(["multiclass_IRA_kappas", "per_class_IRA_kappas"], keys(plot_data))
        IRA_kappa_data = multiclass ?
                         vcat(plot_data["multiclass_IRA_kappas"], plot_data["per_class_IRA_kappas"]) :
                         plot_data["per_class_IRA_kappas"]
    end

    kappa = plot_kappas!(fig[1, 3], kappa_data, labels, IRA_kappa_data; annotation_text_size=textsize)

    # Curves
    plot_pr_curves!(fig[2, 1], plot_data["per_class_pr_curves"], plot_data["class_labels"]; legend=nothing)

    plot_prg_curves!(fig[2, 2], plot_data["per_class_prg_curves"], plot_data["per_class_prg_aucs"],
                     plot_data["class_labels"]; legend=nothing)

    ax = plot_roc_curves!(fig[2, 3], plot_data["per_class_roc_curves"], plot_data["per_class_roc_aucs"],
                          plot_data["class_labels"]; legend=nothing)

    plot_reliability_calibration_curves!(fig[3, 1], plot_data["per_class_reliability_calibration_curves"],
                                         plot_data["per_class_reliability_calibration_scores"],
                                         plot_data["class_labels"]; legend=nothing)

    legend_pos = 2:3
    if haskey(plot_data, "discrimination_calibration_curve")
        legend_pos = 3
        curve = plot_binary_discrimination_calibration_curves!(fig[3, 2],
                                                               plot_data["discrimination_calibration_curve"],
                                                               plot_data["discrimination_calibration_score"],
                                                               plot_data["per_expert_discrimination_calibration_curves"],
                                                               plot_data["per_expert_discrimination_calibration_scores"],
                                                               plot_data["optimal_threshold"],
                                                               plot_data["class_labels"][plot_data["optimal_threshold_class"]])
    end
    elements = map(AbstractPlotting.MakieLayout.legendelements(ax.scene)) do elem
        return [PolyElement(; color=elem.color, strokecolor=:transparent)]
    end

    function label_str(i)
        auc = round(plot_data["per_class_roc_aucs"][i]; digits=2)
        mse = round(plot_data["per_class_reliability_calibration_scores"][i]; digits=2)
        return ["""ROC AUC  $auc
                   Cal. MSE    $mse
                   """]
    end
    classes = plot_data["class_labels"]
    nclasses = length(classes)
    class_labels = label_str.(1:nclasses)
    Legend(fig[3, legend_pos], elements, class_labels, classes; nbanks=2,
           tellwidth=false, tellheight=false,
           labelsize=11, titlegap=5, groupgap=6, labelhalign=:left,
           labelvalign=:center)
    colgap!(fig.layout, 3)
    return fig
end


# Hack until we make a better integration with the recipe system and have these defined automatically
for name in (:plot_reliability_calibration_curves, :plot_prg_curves, :plot_pr_curves,
                  :plot_roc_curves, :plot_kappas, :plot_confusion_matrix,
                  :evaluation_metrics_plot)
    @eval begin
        function $(name)(args...; resolution=(600, 800), plot_kw...)
            fig = Figure(resolution=resolution)
            ax = $(Symbol(string(name, "!")))(fig[1, 1], args...; plot_kw...)
            # ax.plots[1] is not really that great, but there isn't a FigureAxis object right now
            # this will need to wait for when we figure out a better recipe integration
            return AbstractPlotting.FigureAxisPlot(fig, ax, ax.scene.plots[1])
        end
    end
end

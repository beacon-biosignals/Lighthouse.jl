# Confusion matrices

```@docs
Lighthouse.plot_confusion_matrix
```

```@setup 1
using Lighthouse
using CairoMakie
CairoMakie.activate!(type = "png")
using StableRNGs
const RNG = StableRNG(22)
stable_rand(args...) = rand(RNG, args...)
stable_randn(args...) = randn(RNG, args...)
```

```@example 1
using Lighthouse: plot_confusion_matrix, plot_confusion_matrix!

classes = ["red", "orange", "yellow", "green"]
ground_truth =     [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
predicted_labels = [1, 1, 1, 1, 2, 2, 4, 4, 4, 4, 4, 3]
confusion = Lighthouse.confusion_matrix(length(classes), zip(predicted_labels, ground_truth))

fig, ax, p = plot_confusion_matrix(confusion, classes)
```

```@example 1
fig = Figure(size=(800, 400))
plot_confusion_matrix!(fig[1, 1], confusion, classes, :Row, annotation_text_size=14)
plot_confusion_matrix!(fig[1, 2], confusion, classes, :Column, annotation_text_size=14)
fig
```

## Theming

All plots are globally themeable, by setting their `camelcase(functionname)` to a theme. Usually, there are a few sub categories, for e.g. axis, text and subplots.

!!! warning
    Make sure, that you spell names correctly and fully construct the named tuples in the calls. E.g. `(color=:red)` is _not_ a named tuple - it needs to be `(color=:red,)`. Misspelled names and badly constructed named tuples are not easy to error on, since those theming attributes are global, and may be valid for other plots.

```@example 1
with_theme(
        ConfusionMatrix = (
            Text = (
                color=:yellow,
            ),
            Heatmap = (
                colormap=:greens,
            ),
            Axis = (
                backgroundcolor=:black,
                xticklabelrotation=0.0,
            )
        )
    ) do
    plot_confusion_matrix(confusion, classes, :Row)
end
```

# Reliability calibration curves

```@docs
Lighthouse.plot_reliability_calibration_curves
```

```@example 1
using Lighthouse: plot_reliability_calibration_curves
classes = ["class $i" for i in 1:5]
curves = [(LinRange(0, 1, 10), range(0, stop=i/2, length=10) .+ (stable_randn(10) .* 0.1)) for i in -1:3]

plot_reliability_calibration_curves(
    curves,
    stable_rand(5),
    classes
)
```

Note that all curve plot types accepts these types:

```@docs
Lighthouse.XYVector
Lighthouse.SeriesCurves
```

## Theming

All generic series and axis attributes can be themed via `SeriesPlot.Series` / `SeriesPlot.Axis`.
You can have a look at [the series doc to get an idea about the applicable attributes](http://makie.juliaplots.org/stable/plotting_functions/series.html).
To style specifics of a subplot inside the curve plot, e.g. the ideal lineplot, one can use the camel case function name (without `plot_`) and pass those attributes there.
So e.g the `ideal` curve inside the reliability curve can be themed like this:
```@example 1
# The axis is getting created in the seriesplot,
# to always have these kind of probabilistic series have the same axis
series_theme = (
    Axis = (
        backgroundcolor = (:gray, 0.1),
        bottomspinevisible = false,
        leftspinevisible = false,
        topspinevisible = false,
        rightspinevisible = false,
    ),
    Series = (
        color=:darktest,
        marker=:circle
    )
)
with_theme(
        ReliabilityCalibrationCurves = (
            Ideal = (
                color=:red, linewidth=3
            ),
        ),
        SeriesPlot = series_theme
    ) do
    plot_reliability_calibration_curves(
        curves,
        stable_rand(5),
        classes
    )
end
```

# Binary Discrimination Calibration Curves

```@docs
Lighthouse.plot_binary_discrimination_calibration_curves
```

```@example 1
using Lighthouse: plot_binary_discrimination_calibration_curves

Lighthouse.plot_binary_discrimination_calibration_curves(
    curves[3],
    stable_rand(5),
    curves[[1, 2, 4, 5]],
    nothing, nothing,
    "",
)
```

# PR curves

```@docs
Lighthouse.plot_pr_curves
```

```@example 1
using Lighthouse: plot_pr_curves
plot_pr_curves(
    curves,
    classes
)
```

## Theming

```@example 1
# The plots with only a series don't have a special keyword
with_theme(SeriesPlot = series_theme) do
    plot_pr_curves(
        curves,
        classes
    )
end
```


# ROC curves

```@docs
Lighthouse.plot_roc_curves
```

```@example 1
using Lighthouse: plot_roc_curves

plot_roc_curves(
    curves,
    stable_rand(5),
    classes,
    legend=:lt)
```


## Theming

```@example 1
# The plots with only a series don't have a special keyword
with_theme(SeriesPlot = series_theme) do
    plot_roc_curves(
        curves,
        stable_rand(5),
        classes,
        legend=:lt)
end
```

# Kappas (per expert agreement)

```@docs
Lighthouse.plot_kappas
```

```@example 1
using Lighthouse: plot_kappas
plot_kappas(stable_rand(5), classes)
```

```@example 1
using Lighthouse: plot_kappas
plot_kappas(stable_rand(5), classes, stable_rand(5))
```
## Theming

```@example 1
with_theme(
        Kappas = (
            Axis = (
                xticklabelsvisible=false,
                xticksvisible=false,
                leftspinevisible = false,
                rightspinevisible = false,
                bottomspinevisible = false,
                topspinevisible = false,
            ),
            Text = (
                color = :blue,
            ),
            BarPlot = (color=[:black, :green],)
        )) do
    plot_kappas((1:5) ./ 5 .- 0.1, classes, (1:5) ./ 5)
end
```

# Evaluation metrics plot

```@docs
Lighthouse.evaluation_metrics_plot
```

```@example 1
using Lighthouse: evaluation_metrics_plot
data = Dict{String, Any}()
data["confusion_matrix"] = stable_rand(0:100, 5, 5)
data["class_labels"] = classes

data["per_class_kappas"] = stable_rand(5)
data["multiclass_kappa"] = stable_rand()
data["per_class_IRA_kappas"] = stable_rand(5)
data["multiclass_IRA_kappas"] = stable_rand()

data["per_class_pr_curves"] = curves
data["per_class_roc_curves"] = curves
data["per_class_roc_aucs"] = stable_rand(5)

data["per_class_reliability_calibration_curves"] = curves
data["per_class_reliability_calibration_scores"] = stable_rand(5)

evaluation_metrics_plot(data)
```

Optionally, one can also add a binary discrimination calibration curve plot:

```@example 1
data["discrimination_calibration_curve"] = (LinRange(0, 1, 10), LinRange(0,1, 10) .+ 0.1randn(10))
data["per_expert_discrimination_calibration_curves"] = curves

# These are currently not used in plotting, but are still passed to `plot_binary_discrimination_calibration_curves`!
data["discrimination_calibration_score"] = missing
data["optimal_threshold_class"] = 1
data["per_expert_discrimination_calibration_scores"] = missing
data["optimal_threshold"] = missing

evaluation_metrics_plot(data)
```

Plots can also be generated directly from an `EvaluationV1`:
```@example 1
data_row = EvaluationV1(data)
evaluation_metrics_plot(data_row)
```

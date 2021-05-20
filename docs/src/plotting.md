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

fig, ax, p = plot_confusion_matrix(confusion, classes, :Row)
```

```@example 1
fig = Figure(resolution=(800, 400))
plot_confusion_matrix!(fig[1, 1], confusion, classes, :Row, annotation_text_size=14)
plot_confusion_matrix!(fig[1, 2], confusion, classes, :Column, annotation_text_size=14)
fig
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

# PRG curves

```@docs
Lighthouse.plot_prg_curves
```

```@example 1
using Lighthouse: plot_prg_curves
plot_prg_curves(
    curves,
    stable_rand(5),
    classes
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

# Evaluation metrics plot

```@docs
Lighthouse.evaluation_metrics_plot
```

```@example 1
using Lighthouse: evaluation_metrics_plot
data = Dict{String, Any}()
data["confusion_matrix"] = stable_rand(5, 5)
data["class_labels"] = classes

data["per_class_kappas"] = stable_rand(5)
data["multiclass_kappa"] = stable_rand()
data["per_class_IRA_kappas"] = stable_rand(5)
data["multiclass_IRA_kappas"] = stable_rand()

data["per_class_pr_curves"] = curves
data["per_class_prg_curves"] = curves
data["per_class_roc_curves"] = curves
data["per_class_roc_aucs"] = stable_rand(5)
data["per_class_prg_aucs"] = stable_rand(5)

data["per_class_reliability_calibration_curves"] = curves
data["per_class_reliability_calibration_scores"] = stable_rand(5)

evaluation_metrics_plot(data)
```

Optionally, one can also add a binary discrimination calibration curve plot:

```@example 1

data["discrimination_calibration_curve"] = (LinRange(0, 1, 10), LinRange(0,1, 10) .+ 0.1randn(10))
data["per_expert_discrimination_calibration_curves"] = curves

# These are currently not used in plotting, but are still passed to `plot_binary_discrimination_calibration_curves`!
data["discrimination_calibration_score"] = nothing
data["optimal_threshold_class"] = 1
data["per_expert_discrimination_calibration_scores"] = nothing
data["optimal_threshold"] = nothing

evaluation_metrics_plot(data)
```

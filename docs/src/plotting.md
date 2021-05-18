# Confusion matrices

```@docs
Lighthouse.plot_confusion_matrix
```

```@setup 1
using CairoMakie
CairoMakie.activate!(type = "png")

```

```@example 1
using Lighthouse: plot_confusion_matrix, plot_confusion_matrix!
confusion = [0.1 0.2 0.3 0.4 0.0;
        0.1 0.2 0.3 0.4 0.5;
        0.1 0.2 1.0 0.4 0.5;
        0.1 0.2 0.3 0.4 0.5;
        0.9 0.2 0.3 0.4 0.5]
classes = ["class $i" for i in 1:size(confusion, 1)]
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

curves = [(LinRange(0, 1, 10), range(0, stop=i/2, length=10) .+ (randn(10) .* 0.1)) for i in -1:3]

plot_reliability_calibration_curves(
    curves,
    rand(5),
    classes
)
```

# PRG curves

```@docs
Lighthouse.plot_prg_curves
```

```@example 1
using Lighthouse: plot_prg_curves
plot_prg_curves(
    curves,
    rand(5),
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
    rand(5),
    classes,
    legend=:lt)
```

# Kappas (per expert agreement)

```@docs
Lighthouse.plot_kappas
```

```@example 1
using Lighthouse: plot_kappas
plot_kappas(rand(5), classes)
```

```@example 1
using Lighthouse: plot_kappas
plot_kappas(rand(5), classes, rand(5))
```

# Evaluation metrics plot

```@docs
Lighthouse.evaluation_metrics_plot
```

```@example 1
using Lighthouse: evaluation_metrics_plot
data = Dict{String, Any}()
data["confusion_matrix"] = confusion
data["class_labels"] = classes

data["per_class_kappas"] = rand(5)
data["multiclass_kappa"] = rand()
data["per_class_IRA_kappas"] = rand(5)
data["multiclass_IRA_kappas"] = rand()

data["per_class_pr_curves"] = curves
data["per_class_prg_curves"] = curves
data["per_class_roc_curves"] = curves
data["per_class_roc_aucs"] = rand(5)
data["per_class_prg_aucs"] = rand(5)

data["per_class_reliability_calibration_curves"] = curves
data["per_class_reliability_calibration_scores"] = rand(5)

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

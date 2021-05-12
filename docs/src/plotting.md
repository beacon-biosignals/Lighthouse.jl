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
data = [0.1 0.2 0.3 0.4 0.0;
        0.1 0.2 0.3 0.4 0.5;
        0.1 0.2 1.0 0.4 0.5;
        0.1 0.2 0.3 0.4 0.5;
        0.9 0.2 0.3 0.4 0.5]
classes = ["class $i" for i in 1:size(data, 1)]
fig, ax, p = plot_confusion_matrix(data, classes, :Row)
```

```@example 1
fig = Figure(resolution=(800, 400))
plot_confusion_matrix!(fig[1, 1], data, classes, :Row)
plot_confusion_matrix!(fig[1, 2], data, classes, :Column)
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
    classes)
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

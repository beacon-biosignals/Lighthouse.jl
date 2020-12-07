# API Documentation

```@meta
CurrentModule = Lighthouse
```

## The `AbstractClassifier` Interface

```@docs
AbstractClassifier
Lighthouse.classes
Lighthouse.train!
Lighthouse.loss_and_prediction
Lighthouse.onehot
Lighthouse.onecold
Lighthouse.is_early_stopping_exception
```

## The `learn!` Interface

```@docs
LearnLogger
learn!
upon
evaluate!
predict!
Lighthouse.forward_logs
Lighthouse._calculate_ea_kappas
Lighthouse._calculate_ira_kappas
Lighthouse._calculate_spearman_correlation
Lighthouse.evaluation_metrics_plot
Base.flush
```

## Performance Metrics

```@docs
confusion_matrix
accuracy
binary_statistics
cohens_kappa
calibration_curve
```

## Utilities

```@docs
majority
Lighthouse.area_under_curve
Lighthouse.area_under_curve_unit_square
```

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
learn!
evaluate!
predict!
Lighthouse._calculate_ea_kappas
Lighthouse._calculate_ira_kappas
Lighthouse._calculate_spearman_correlation
```

## The logging interface

The following "primitives" must be defined for a logger to be used with Lighthouse:

```@docs
log_event!
log_value!
log_line_series!
log_plot!
step_logger!
summarize_array
```

These primitives can be used in implementations of [`train!`](@ref), [`evaluate!`](@ref), and [`predict!`](@ref), as well as in:

```@docs
Lighthouse.log_evaluation_row!
log_values!
```

### `LearnLogger`s

`LearnLoggers` are a Tensorboard-backed logger which comply with the above logging interface. They also support additional callback functionality with `upon`:

```@docs
LearnLogger
upon
Lighthouse.forward_logs
```

## Performance Metrics

```@docs
confusion_matrix
accuracy
binary_statistics
cohens_kappa
calibration_curve
EvaluationRow
ObservationRow
Lighthouse.evaluation_metrics
Lighthouse._evaluation_row_dict
Lighthouse.evaluation_metrics_row
```

## Utilities

```@docs
majority
Lighthouse.area_under_curve
Lighthouse.area_under_curve_unit_square
flush(::LearnLogger)
```

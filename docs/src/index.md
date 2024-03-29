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
```

## The logging interface

The following "primitives" must be defined for a logger to be used with Lighthouse:

```@docs
log_value!
log_line_series!
log_plot!
step_logger!
```

in addition to `Base.flush(logger)` (which can be a no-op by defining `Base.flush(::MyLoggingType) = nothing`).

These primitives can be used in implementations of [`train!`](@ref), [`evaluate!`](@ref), and [`predict!`](@ref), as well as in the following composite logging functions, which by default call the above primitives. Loggers may provide custom implementations of these.

```@docs
log_event!
Lighthouse.log_evaluation_row!
log_values!
log_array!
log_arrays!
```

### `LearnLogger`s

`LearnLoggers` are a Tensorboard-backed logger which comply with the above logging interface. They also support additional callback functionality with `upon`:

```@docs
LearnLogger
upon
Lighthouse.forward_logs
flush(::LearnLogger)
```

## Performance Metrics

```@docs
confusion_matrix
accuracy
binary_statistics
cohens_kappa
calibration_curve
EvaluationV1
ObservationV1
Lighthouse.evaluation_metrics
Lighthouse._evaluation_dict
Lighthouse.evaluation_metrics_record
Lighthouse.ClassV1
TradeoffMetricsV1
get_tradeoff_metrics
get_tradeoff_metrics_binary_multirater
HardenedMetricsV1
get_hardened_metrics
get_hardened_metrics_multirater
get_hardened_metrics_multiclass
LabelMetricsV1
get_label_metrics_multirater
get_label_metrics_multirater_multiclass
Lighthouse._evaluation_record
Lighthouse._calculate_ea_kappas
Lighthouse._calculate_ira_kappas
Lighthouse._calculate_spearman_correlation
```

## Utilities

```@docs
majority
Lighthouse.area_under_curve
Lighthouse.area_under_curve_unit_square
Curve
```

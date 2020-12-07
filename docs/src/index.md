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
```

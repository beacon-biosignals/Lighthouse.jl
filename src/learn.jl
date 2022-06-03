#####
##### Logging interface
#####

# These must be implemented by every logger type.

"""
    log_plot!(logger, field::AbstractString, plot, plot_data)

Log a `plot` to `logger` under field `field`.

* `plot`: the plot itself
* `plot_data`: an unstructured dictionary of values used in creating `plot`.

See also [`log_line_series!`](@ref).
"""
log_plot!(logger, field::AbstractString, plot, plot_data)

"""
    log_value!(logger, field::AbstractString, value)

Log a value `value` to `field`.
"""
log_value!(logger, field::AbstractString, value)

"""
    log_line_series!(logger, field::AbstractString, curves, labels=1:length(curves))

Logs a series plot to `logger` under `field`, where...

- `curves` is an iterable of the form `Tuple{Vector{Real},Vector{Real}}`, where each tuple contains `(x-values, y-values)`, as in the `Lighthouse.EvaluationRow` field `per_class_roc_curves`
- `labels` is the class label for each curve, which defaults to the numeric index of each curve.
"""
log_line_series!(logger, field::AbstractString, curves, labels=1:length(curves))


# The following have default implementations.

"""
    step_logger!(logger)

Increments the `logger`'s `step`, if any. Defaults to doing nothing.
"""
step_logger!(::Any) = nothing

"""
    log_event!(logger, value::AbstractString)

Logs a string event given by `value` to `logger`. Defaults to calling `log_value!` with a field named `event`.
"""
function log_event!(logger, value::AbstractString)
    return log_value!(logger, "event", string(now(), " | ", value))
end

"""
    log_values!(logger, values)

Logs an iterable of `(field, value)` pairs to `logger`. Falls back to calling `log_value!` in a loop.
Loggers may specialize this method for improved performance.
"""
function log_values!(logger, values)
    for (k, v) in values
        log_value!(logger, k, v)
    end
    return nothing
end

function log_numeric_and_string!(logger, field::AbstractString, values)
    values = filter!((k, v) -> isa(v, Union{Number, AbstractString}), values)
    values = map((k, v) -> (string(field, k), v), values)
    log_values!(logger,values)
    return nothing
end

"""
    log_hardened_row!!(logger, field::AbstractString, metrics)

Log a hardened row `metrics` to `field`.
"""
log_hardened_row!(logger, field::AbstractString, metrics) = log_numeric_and_string!(logger, pairs(metrics))
"""
    log_tradeoff_row!!(logger, field::AbstractString, metrics)

Log a tradeoff row `metrics` to `field`.
"""
log_tradeoff_row!(logger, field::AbstractString, metrics) = log_numeric_and_string!(logger, pairs(metrics))
"""
    log_labels_row!!(logger, field::AbstractString, metrics)

Log a labels row `metrics` to `field`.
"""
log_labels_row!(logger, field::AbstractString, metrics) = log_numeric_and_string!(logger, pairs(metrics))

"""
    log_array!(logger::Any, field::AbstractString, value)

Log an array `value` to `field`.

Defaults to `log_value!(logger, mean(value))`.
"""
function log_array!(logger::Any, field::AbstractString, array)
    return log_value!(logger, field, mean(array))
end

"""
    log_arrays!(logger, values)

Logs an iterable of `(field, array)` pairs to `logger`. Falls back to calling `log_array!` in a loop.

Loggers may specialize this method for improved performance.
"""
function log_arrays!(logger, values)
    for (k, v) in values
        log_array!(logger, k, v)
    end
    return nothing
end


"""
    log_evaluation_row!(logger, field::AbstractString, metrics)

From fields in [`EvaluationRow`](@ref), generate and plot the composite [`evaluation_metrics_plot`](@ref)
as well as `spearman_correlation` (if present).
"""
function log_evaluation_row!(logger, field::AbstractString, metrics)
    metrics_plot = evaluation_metrics_plot(metrics)
    metrics_dict = _evaluation_row_dict(metrics)
    log_plot!(logger, field, metrics_plot, metrics_dict)
    if haskey(metrics_dict, "spearman_correlation")
        sp_field = replace(field, "metrics" => "spearman_correlation")
        log_value!(logger, sp_field, metrics_dict["spearman_correlation"].ρ)
    end
    return metrics_plot
end

function log_resource_info!(logger, section::AbstractString, info::ResourceInfo;
                            suffix::AbstractString="")
    log_values!(logger,
                (section * "/time_in_seconds" * suffix => info.time_in_seconds,
                 section * "/gc_time_in_seconds" * suffix => info.gc_time_in_seconds,
                 section * "/allocations" * suffix => info.allocations,
                 section * "/memory_in_mb" * suffix => info.memory_in_mb))
    return info
end

function log_resource_info!(f, logger, section::AbstractString; suffix::AbstractString="")
    result, resource_info = call_with_resource_info(f)
    log_resource_info!(logger, section, resource_info; suffix=suffix)
    return result
end

#####
##### `predict!!`
#####

"""
    predict!(model::AbstractClassifier,
             predicted_soft_labels::AbstractMatrix,
             batches, logger::LearnLogger;
             logger_prefix::AbstractString)

Return `mean_loss` of all `batches` after using `model` to predict their soft labels
and storing those results in `predicted_soft_labels`.

The following quantities are logged to `logger`:
 - `<logger_prefix>/loss_per_batch`
 - `<logger_prefix>/mean_loss_per_epoch`
 - `<logger_prefix>/\$resource_per_batch`

Where...

- `model` is a model that outputs soft labels when called on a batch of `batches`,
  `model(batch)`.

- `predicted_soft_labels` is a matrix whose columns correspond to classes and
  whose rows correspond to samples in batches, and which is filled in with soft-label
  predictions.

- `batches` is an iterable of batches, where each element of
  the iterable takes the form `(batch, votes_locations)`. Internally, `batch` is
  passed to [`loss_and_prediction`](@ref) as `loss_and_prediction(model, batch...)`.

 """
function predict!(model::AbstractClassifier, predicted_soft_labels::AbstractMatrix, batches,
                  logger; logger_prefix::AbstractString)
    losses = Float32[]
    for (batch, votes_locations) in batches
        batch_loss = log_resource_info!(logger, logger_prefix; suffix="_per_batch") do
            batch_loss, soft_label_batch = loss_and_prediction(model, batch...)
            for (i, soft_label) in enumerate(eachcol(soft_label_batch))
                predicted_soft_labels[votes_locations[i], :] = soft_label
            end
            return batch_loss
        end
        log_value!(logger, logger_prefix * "/loss_per_batch", batch_loss)
        push!(losses, batch_loss)
    end
    mean_loss = mean(losses)
    log_value!(logger, logger_prefix * "/mean_loss_per_epoch", mean_loss)
    return mean_loss
end

#####
##### `evaluate!`
#####

"""
    evaluate!(predicted_hard_labels::AbstractVector,
              predicted_soft_labels::AbstractMatrix,
              elected_hard_labels::AbstractVector,
              classes, logger;
              logger_prefix, logger_suffix,
              votes::Union{Nothing,AbstractMatrix}=nothing,
              thresholds=0.0:0.01:1.0,
              optimal_threshold_class::Union{Nothing,Integer}=nothing)

Return `nothing` after computing and logging a battery of classifier performance
metrics that each compare `predicted_soft_labels` and/or `predicted_hard_labels`
agaist `elected_hard_labels`.

The following quantities are logged to `logger`:
    - `<logger_prefix>/metrics<logger_suffix>`
    - `<logger_prefix>/\$resource<logger_suffix>`

Where...

- `predicted_soft_labels` is a matrix of soft labels whose columns correspond to
  classes and whose rows correspond to samples in the evaluation set.

- `predicted_hard_labels` is a vector of hard labels where the `i`th element
  is the hard label predicted by the model for sample `i` in the evaulation set.

- `elected_hard_labels` is a vector of hard labels where the `i`th element
  is the hard label elected as "ground truth" for sample `i` in the evaulation set.

- `thresholds` are the range of thresholds used by metrics (e.g. PR curves) that
  are calculated on the `predicted_soft_labels` for a range of thresholds.

- `votes` is a matrix of hard labels whose columns correspond to voters and whose
  rows correspond to the samples in the test set that have been voted on. If
  `votes[sample, voter]` is not a valid hard label for `model`, then `voter` will
  simply be considered to have not assigned a hard label to `sample`.

- `optimal_threshold_class` is the class index (`1` or `2`) for which to calculate
  an optimal threshold for converting the `predicted_soft_labels` to
  `predicted_hard_labels`. If present, the input `predicted_hard_labels` will be
  ignored and new `predicted_hard_labels` will be recalculated from the new threshold.
  This is only a valid parameter when `length(classes) == 2`
"""
function evaluate!(predicted_hard_labels::AbstractVector,
                   predicted_soft_labels::AbstractMatrix,
                   elected_hard_labels::AbstractVector, classes, logger;
                   logger_prefix::AbstractString, logger_suffix::AbstractString="",
                   votes::Union{Nothing,Missing,AbstractMatrix}=nothing,
                   thresholds=0.0:0.01:1.0,
                   optimal_threshold_class::Union{Nothing,Integer,Missing}=nothing)
    _validate_threshold_class(optimal_threshold_class, classes)

    log_resource_info!(logger, logger_prefix; suffix=logger_suffix) do
        metrics = evaluation_metrics_row(predicted_hard_labels, predicted_soft_labels,
                                         elected_hard_labels, classes, thresholds;
                                         votes, optimal_threshold_class)
        log_evaluation_row!(logger, logger_prefix * "/metrics" * logger_suffix,
                            metrics)
        return nothing
    end
    return nothing
end

#####
##### `learn!`
#####

"""
    learn!(model::AbstractClassifier, logger,
           get_train_batches, get_test_batches, votes,
           elected=majority.(eachrow(votes), (1:length(classes(model)),));
           epoch_limit=100, post_epoch_callback=(_ -> nothing),
           optimal_threshold_class::Union{Nothing,Integer}=nothing,
           test_set_logger_prefix="test_set")

Return `model` after optimizing its parameters across multiple epochs of
training and test, logging Lighthouse's standardized suite of classifier
performance metrics to `logger` throughout the optimization process.

The following phases are executed at each epoch (note: in the below lists
of logged values, `\$resource` takes the values of the field names of
`Lighthouse.ResourceInfo`):

1. Train `model` by calling `train!(model, get_train_batches(), logger)`.
   The following quantities are logged to `logger` during this phase:
    - `train/loss_per_batch`
    - any additional quantities logged by the relevant model/framework-specific
      implementation of `train!`.

2. Compute `model`'s predictions on test set provided by `get_test_batches()`
   (see below for details). The following quantities are logged to `logger`
   during this phase:
    - `<test_set_logger_prefix>_prediction/loss_per_batch`
    - `<test_set_logger_prefix>_prediction/mean_loss_per_epoch`
    - `<test_set_logger_prefix>_prediction/\$resource_per_batch`

3. Compute a battery of metrics to evaluate `model`'s performance on the test
   set based on the test set prediction phase. The following quantities are
   logged to `logger` during this phase:
    - `<test_set_logger_prefix>_evaluation/metrics_per_epoch`
    - `<test_set_logger_prefix>_evaluation/\$resource_per_epoch`

4. Call `post_epoch_callback(current_epoch)`.

Where...

- `get_train_batches` is a zero-argument function that returns an iterable of
  training set batches. Internally, `learn!` uses this function when it calls
  `train!(model, get_train_batches(), logger)`.

- `get_test_batches` is a zero-argument function that returns an iterable
  of test set batches used during the current epoch's test phase. Each element of
  the iterable takes the form `(batch, votes_locations)`. Internally, `batch` is
  passed to [`loss_and_prediction`](@ref) as `loss_and_prediction(model, batch...)`,
  and `votes_locations[i]` is expected to yield the row index of `votes` that
  corresponds to the `i`th sample in `batch`.

- `votes` is a matrix of hard labels whose columns correspond to voters and whose
  rows correspond to the samples in the test set that have been voted on. If
  `votes[sample, voter]` is not a valid hard label for `model`, then `voter` will
  simply be considered to have not assigned a hard label to `sample`.

- `elected` is a vector of hard labels where the `i`th element is the hard label
  elected as "ground truth" out of `votes[i, :]`.

- `optimal_threshold_class` is the class index (`1` or `2`) for which to calculate
  an optimal threshold for converting `predicted_soft_labels` to `predicted_hard_labels`.
  This is only a valid parameter when `length(classes) == 2`. If `optimal_threshold_class`
  is present, test set evaluation will be based on predicted hard labels calculated
  with this threshold; if `optimal_threshold_class` is `nothing`, predicted hard labels
  will be calculated via `onecold(classifier, soft_label)`.
"""
function learn!(model::AbstractClassifier, logger, get_train_batches, get_test_batches,
                votes, elected=majority.(eachrow(votes), (1:length(classes(model)),));
                epoch_limit=100, post_epoch_callback=(_ -> nothing),
                optimal_threshold_class::Union{Nothing,Integer}=nothing,
                test_set_logger_prefix="test_set")
    # NOTE `votes` is currently unused except to construct `elected` by default,
    # but will be necessary for calculating multirater metrics e.g. Fleiss' kappa
    # later so we still required it in the API
    # TODO keeping `votes` alive might hog a lot of memory, so it would be convenient
    # to provide callers with a wrapper around `channel_unordered` (or at least example
    # code) for generating an iterator that batches `votes` rows to `soft labels` in
    # a manner that's respectful to throughput/memory constraints.
    # TODO is it better to pre-allocate these, or allocate them per-epoch? The
    # former ensures fixed memory usage, but the latter gives the GC a chance
    # to free up some RAM between epochs.
    _validate_threshold_class(optimal_threshold_class, classes(model))

    predicted = zeros(Float32, length(elected), length(classes(model)))
    log_event!(logger, "started learning")
    for current_epoch in 1:epoch_limit
        try
            train!(model, get_train_batches(), logger)
            predict!(model, predicted, get_test_batches(), logger;
                     logger_prefix="$(test_set_logger_prefix)_prediction")
            evaluate!(map(label -> onecold(model, label), eachrow(predicted)), predicted,
                      elected, classes(model), logger;
                      logger_prefix="$(test_set_logger_prefix)_evaluation",
                      logger_suffix="_per_epoch", votes=votes,
                      optimal_threshold_class=optimal_threshold_class)
            post_epoch_callback(current_epoch)
            flush(logger)
        catch ex # support early stopping via exception handling
            if is_early_stopping_exception(model, ex)
                log_event!(logger, "`learn!` call stopped via $ex in epoch $current_epoch")
                break
            else
                log_event!(logger,
                           "`learn!` call encountered exception in epoch $current_epoch")
                rethrow(ex)
            end
        end
    end
    return model
end

#####
##### `post_epoch_callback` utilities
#####

"""
    upon(logger::LearnLogger, field::AbstractString; condition, initial)
    upon(logged::Dict{String,Any}, field::AbstractString; condition, initial)

Return a closure that can be called to check the most recent state of
`logger.logged[field]` and trigger a caller-provided function when
`condition(recent_state, previously_chosen_state)` is `true`.

For example:

```
upon_loss_decrease = upon(logger, "test_set_prediction/mean_loss_per_epoch";
                          condition=<, initial=Inf)

save_upon_loss_decrease = _ -> begin
    upon_loss_decrease(new_lowest_loss -> save_my_model(model, new_lowest_loss),
                       consecutive_failures -> consecutive_failures > 10 && Flux.stop())
end

learn!(model, logger, get_train_batches, get_test_batches, votes;
       post_epoch_callback=save_upon_loss_decrease)
```

Specifically, the form of the returned closure is `f(on_true, on_false)` where
`on_true(state)` is called if `condition(state, previously_chosen_state)` is
`true`. Otherwise, `on_false(consecutive_falses)` is called where `consecutive_falses`
is the number of `condition` calls that have returned `false` since the last
`condition` call returned `true`.

Note that the returned closure is a no-op if `logger.logged[field]` has not
been updated since the most recent call.
"""
function upon(logged::Dict{String,Vector{Any}}, field::AbstractString; condition, initial)
    history = get!(() -> Any[], logged, field)
    previous_length = length(history)
    current = isempty(history) ? initial : last(history)
    consecutive_false_count = 0
    return (on_true, on_false=(_ -> nothing)) -> begin
        length(history) == previous_length && return nothing
        previous_length = length(history)
        candidate = last(history)
        if condition(candidate, current)
            consecutive_false_count = 0
            current = candidate
            on_true(current)
        else
            consecutive_false_count += 1
            on_false(consecutive_false_count)
        end
    end
end

function upon(logger::LearnLogger, field::AbstractString; condition, initial)
    return upon(logger.logged, field; condition=condition, initial=initial)
end

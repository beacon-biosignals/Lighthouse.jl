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
log_line_series!(logger, field::AbstractString, curves; labels=1:length(curves))

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

function _calculate_stratified_ea_kappas(predicted_hard_labels, elected_hard_labels,
                                         class_count, strata)
    groups = reduce(∪, strata)
    kappas = Pair{String,Any}[]
    for group in groups
        index = group .∈ strata
        predicted = predicted_hard_labels[index]
        elected = elected_hard_labels[index]
        k = _calculate_ea_kappas(predicted, elected, class_count)
        push!(kappas,
              group => (per_class=k.per_class_kappas, multiclass=k.multiclass_kappa,
                        n=sum(index)))
    end
    kappas = sort(kappas; by=p -> last(p).multiclass)
    return [k = v for (k, v) in kappas]
end

"""
    _calculate_ea_kappas(predicted_hard_labels, elected_hard_labels, classes)

Return `NamedTuple` with keys `:per_class_kappas`, `:multiclass_kappa` containing the Cohen's
Kappa per-class and over all classes, respectively. The value of output key
`:per_class_kappas` is an `Array` such that item `i` is the Cohen's kappa calculated
for class `i`.

Where...

- `predicted_hard_labels` is a vector of hard labels where the `i`th element
  is the hard label predicted by the model for sample `i` in the evaulation set.

- `elected_hard_labels` is a vector of hard labels where the `i`th element
  is the hard label elected as "ground truth" for sample `i` in the evaulation set.

- `class_count` is the number of possible classes.

"""
function _calculate_ea_kappas(predicted_hard_labels, elected_hard_labels, class_count)
    multiclass_kappa = first(cohens_kappa(class_count,
                                          zip(predicted_hard_labels, elected_hard_labels)))

    per_class_kappas = map(1:class_count) do class_index
        return _calculate_ea_kappa(predicted_hard_labels, elected_hard_labels, class_index)
    end
    return (per_class_kappas, multiclass_kappa)
end

function _calculate_ea_kappa(predicted_hard_labels, elected_hard_labels, class_index)
    CLASS_VS_ALL_CLASS_COUNT = 2
    predicted = ((label == class_index) + 1 for label in predicted_hard_labels)
    elected = ((label == class_index) + 1 for label in elected_hard_labels)
    return first(cohens_kappa(CLASS_VS_ALL_CLASS_COUNT, zip(predicted, elected)))
end

has_value(x) = !isnothing(x) && !ismissing(x)

"""
    _calculate_ira_kappas(votes, classes)

Return `NamedTuple` with keys `:per_class_IRA_kappas`, `:multiclass_IRA_kappas` containing the Cohen's
Kappa for inter-rater agreement (IRA) per-class and over all classes, respectively.
The value of output key `:per_class_IRA_kappas` is an `Array` such that item `i` is the
IRA kappa calculated for class `i`.

Where...

- `votes` is a matrix of hard labels whose columns correspond to voters and whose
  rows correspond to the samples in the test set that have been voted on. If
  `votes[sample, voter]` is not a valid hard label for `model`, then `voter` will
  simply be considered to have not assigned a hard label to `sample`.

- `classes` all possible classes voted on.

Returns `(per_class_IRA_kappas=missing, multiclass_IRA_kappas=missing)` if `votes` has only a single voter (i.e., a single column) or if
no two voters rated the same sample. Note that vote entries of `0` are taken to
mean that the voter did not rate that sample.
"""
function _calculate_ira_kappas(votes, classes)
    # no votes given or only one expert:
    if !has_value(votes) || size(votes, 2) < 2
        return (; per_class_IRA_kappas=missing, multiclass_IRA_kappas=missing)
    end

    all_hard_label_pairs = Array{Int}(undef, 0, 2)
    num_voters = size(votes, 2)
    for i_voter in 1:(num_voters - 1)
        for j_voter in (i_voter + 1):num_voters
            all_hard_label_pairs = vcat(all_hard_label_pairs, votes[:, [i_voter, j_voter]])
        end
    end
    hard_label_pairs = filter(row -> all(row .!= 0), collect(eachrow(all_hard_label_pairs)))
    length(hard_label_pairs) > 0 ||
        return (; per_class_IRA_kappas=missing, multiclass_IRA_kappas=missing)  # No common observations voted on
    length(hard_label_pairs) < 10 &&
        @warn "...only $(length(hard_label_pairs)) in common, potentially questionable IRA results"

    multiclass_ira = first(cohens_kappa(length(classes), hard_label_pairs))

    CLASS_VS_ALL_CLASS_COUNT = 2
    per_class_ira = map(1:length(classes)) do class_index
        class_v_other_hard_label_pair = map(row -> 1 .+ (row .== class_index),
                                            hard_label_pairs)
        return first(cohens_kappa(CLASS_VS_ALL_CLASS_COUNT, class_v_other_hard_label_pair))
    end
    return (; per_class_IRA_kappas=per_class_ira, multiclass_IRA_kappas=multiclass_ira)
end

function _spearman_corr(predicted_soft_labels, elected_soft_labels)
    n = length(predicted_soft_labels)
    ρ = StatsBase.corspearman(predicted_soft_labels, elected_soft_labels)
    if isnan(ρ)
        @warn "Uh oh, correlation is NaN! Probably because StatsBase.corspearman(...)
               returns NaN when a set of labels is all the same!"
        # Note: accounted for in https://github.com/JuliaStats/HypothesisTests.jl/pull/53/files;
        # probably not worth implementing here until we need it (at which point maybe
        # it will be ready in HypothesisTests!)
    end

    # 95% confidence interval calculated according to
    # https://stats.stackexchange.com/questions/18887/how-to-calculate-a-confidence-interval-for-spearmans-rank-correlation
    stderr = 1.0 / sqrt(n - 3)
    delta = 1.96 * stderr
    ci_lower = tanh(atanh(ρ) - delta)
    ci_upper = tanh(atanh(ρ) + delta)
    return (ρ=ρ, n=n, ci_lower=round(ci_lower; digits=3),
            ci_upper=round(ci_upper; digits=3))
end

"""
    _calculate_spearman_correlation(predicted_soft_labels, votes, classes)

Return `NamedTuple` with keys `:ρ`, `:n`, `:ci_lower`, and `ci_upper` that are
the Spearman correlation constant ρ and its 95% confidence interval bounds.
Only valid for binary classification problems (i.e., `length(classes) == 2`)

Where...

- `predicted_soft_labels` is a matrix of soft labels whose columns correspond to
  the two classes and whose rows correspond to the samples in the test set that have been
  classified. For a given sample, the two class column values must sum to 1 (i.e.,
  softmax has been applied to the classification output).

- `votes` is a matrix of hard labels whose columns correspond to voters and whose
  rows correspond to the samples in the test set that have been voted on. If
  `votes[sample, voter]` is not a valid hard label for `model`, then `voter` will
  simply be considered to have not assigned a hard label to `sample`. May contain
  a single voter (i.e., a single column).

- `classes` are the two classes voted on.
"""
function _calculate_spearman_correlation(predicted_soft_labels, votes, classes=missing)
    if !ismissing(classes)
        length(classes) > 2 && throw(ArgumentError("Only valid for 2-class problems"))
    end
    if !all(x -> x ≈ 1, sum(predicted_soft_labels; dims=2))
        throw(ArgumentError("Input probabiliities fail softmax assumption"))
    end

    class_index = 1 # Note: Result will be the same whether class 1 or class 2
    elected_soft_labels = Vector{Float64}()
    for sample_votes in eachrow(votes)
        actual_sample_votes = filter(v -> v in Set([1, 2]), sample_votes)
        push!(elected_soft_labels, mean(actual_sample_votes .== class_index))
    end
    return _spearman_corr(predicted_soft_labels[:, class_index], elected_soft_labels)
end

function _calculate_optimal_threshold_from_discrimination_calibration(predicted_soft_labels,
                                                                      votes; thresholds,
                                                                      class_of_interest_index)
    elected_probabilities = _elected_probabilities(votes, class_of_interest_index)
    bin_count = min(size(votes, 2) + 1, 10)
    per_threshold_curves = map(thresholds) do thresh
        return calibration_curve(elected_probabilities,
                                 predicted_soft_labels[:, class_of_interest_index] .>=
                                 thresh; bin_count=bin_count)
    end
    i_min = argmin([c.mean_squared_error for c in per_threshold_curves])
    curve = per_threshold_curves[i_min]
    return (threshold=collect(thresholds)[i_min], mse=curve.mean_squared_error,
            plot_curve_data=(mean.(curve.bins), curve.fractions))
end

function _calculate_discrimination_calibration(predicted_hard_labels, votes;
                                               class_of_interest_index)
    elected_probabilities = _elected_probabilities(votes, class_of_interest_index)
    bin_count = min(size(votes, 2) + 1, 10)
    curve = calibration_curve(elected_probabilities, predicted_hard_labels; bin_count)
    return (mse=curve.mean_squared_error,
            plot_curve_data=(mean.(curve.bins), curve.fractions))
end

function _elected_probabilities(votes, class_of_interest_index)
    elected_probabilities = Vector{Float64}()
    for sample_votes in eachrow(votes)
        actual_sample_votes = filter(v -> v in Set([1, 2]), sample_votes)
        push!(elected_probabilities, mean(actual_sample_votes .== class_of_interest_index))
    end
    return elected_probabilities
end

function _calculate_voter_discrimination_calibration(votes; class_of_interest_index)
    elected_probabilities = _elected_probabilities(votes, class_of_interest_index)

    bin_count = min(size(votes, 2) + 1, 10)
    per_voter_calibration_curves = map(1:size(votes, 2)) do i_voter
        return calibration_curve(elected_probabilities,
                                 votes[:, i_voter] .== class_of_interest_index;
                                 bin_count=bin_count)
    end

    return (mse=map(curve -> curve.mean_squared_error, per_voter_calibration_curves),
            plot_curve_data=map(curve -> (mean.(curve.bins), curve.fractions),
                                per_voter_calibration_curves))
end

function _get_optimal_threshold_from_ROC(roc_curve; thresholds)
    dist = (p1, p2) -> sqrt((p1[1] - p2[1])^2 + (p1[2] - p2[2])^2)
    min = Inf
    curr_counter = 1
    opt_point = nothing
    threshold_idx = 1
    for point in zip(roc_curve[1], roc_curve[2])
        d = dist((0, 1), point)
        if d < min
            min = d
            threshold_idx = curr_counter
            opt_point = point
        end
        curr_counter += 1
    end
    return collect(thresholds)[threshold_idx]
end

function _validate_threshold_class(optimal_threshold_class, classes)
    has_value(optimal_threshold_class) || return nothing
    length(classes) == 2 ||
        throw(ArgumentError("Only valid for binary classification problems"))
    optimal_threshold_class in Set([1, 2]) ||
        throw(ArgumentError("Invalid threshold class"))
    return nothing
end

"""
    evaluation_metrics_row(observation_table, classes, thresholds=0.0:0.01:1.0;
                           strata::Union{Nothing,AbstractVector{Set{T}} where T}=nothing,
                           optimal_threshold_class::Union{Missing,Nothing,Integer}=missing)
    evaluation_metrics_row(predicted_hard_labels::AbstractVector,
                           predicted_soft_labels::AbstractMatrix,
                           elected_hard_labels::AbstractVector,
                           classes,
                           thresholds=0.0:0.01:1.0;
                           votes::Union{Nothing,Missing,AbstractMatrix}=nothing,
                           strata::Union{Nothing,AbstractVector{Set{T}} where T}=nothing,
                           optimal_threshold_class::Union{Missing,Nothing,Integer}=missing)

Returns `EvaluationRow` containing a battery of classifier performance
metrics that each compare `predicted_soft_labels` and/or `predicted_hard_labels`
agaist `elected_hard_labels`.

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

- `strata` is a vector of sets of (arbitrarily typed) groups/strata for each sample
  in the evaluation set, or `nothing`. If not `nothing`, per-class and multiclass
  kappas will also be calculated per group/stratum.

- `optimal_threshold_class` is the class index (`1` or `2`) for which to calculate
  an optimal threshold for converting the `predicted_soft_labels` to
  `predicted_hard_labels`. If present, the input `predicted_hard_labels` will be
  ignored and new `predicted_hard_labels` will be recalculated from the new threshold.
  This is only a valid parameter when `length(classes) == 2`

Alternatively, an `observation_table` that consists of rows of type [`ObservationRow`](@ref)
can be passed in in place of `predicted_soft_labels`,`predicted_hard_labels`,`elected_hard_labels`,
and `votes`.

See also [`evaluation_metrics_plot`](@ref).
"""
function evaluation_metrics_row(observation_table, classes, thresholds=0.0:0.01:1.0;
                                strata::Union{Nothing,AbstractVector{Set{T}} where T}=nothing,
                                optimal_threshold_class::Union{Missing,Nothing,Integer}=missing)
    inputs = _observation_table_to_inputs(observation_table)
    return evaluation_metrics_row(inputs.predicted_hard_labels,
                                  inputs.predicted_soft_labels, inputs.elected_hard_labels,
                                  classes, thresholds; inputs.votes, strata,
                                  optimal_threshold_class)
end

function evaluation_metrics_row(predicted_hard_labels::AbstractVector,
                                predicted_soft_labels::AbstractMatrix,
                                elected_hard_labels::AbstractVector, classes,
                                thresholds=0.0:0.01:1.0;
                                votes::Union{Nothing,Missing,AbstractMatrix}=nothing,
                                strata::Union{Nothing,AbstractVector{Set{T}} where T}=nothing,
                                optimal_threshold_class::Union{Missing,Nothing,Integer}=missing)
    _validate_threshold_class(optimal_threshold_class, classes)

    class_count = length(classes)
    class_vector = collect(classes) # Plots.jl expects this to be an `AbstractVector`
    class_labels = string.(class_vector)
    # per_class_stats = per_class_confusion_statistics(predicted_soft_labels,
    #                                                  elected_hard_labels, thresholds)

    # ROC curves
    # per_class_roc_curves = [(map(t -> t.false_positive_rate, stats),
    #                          map(t -> t.true_positive_rate, stats))
    #                         for stats in per_class_stats]
    # per_class_roc_aucs = [area_under_curve(x, y) for (x, y) in per_class_roc_curves]

    # Optionally calculate optimal threshold
    if has_value(optimal_threshold_class)
        # If votes exist, calculate the threshold based on comparing against
        # vote probabilities. Otherwise, use the ROC curve.
        if has_value(votes)
            # c = _calculate_optimal_threshold_from_discrimination_calibration(predicted_soft_labels,
            #                                                                  votes;
            #                                                                  thresholds=thresholds,
            #                                                                  class_of_interest_index=optimal_threshold_class)
            # optimal_threshold = c.threshold
            discrimination_calibration_curve = c.plot_curve_data
            discrimination_calibration_score = c.mse

            expert_cal = _calculate_voter_discrimination_calibration(votes;
                                                                     class_of_interest_index=optimal_threshold_class)
            per_expert_discrimination_calibration_curves = expert_cal.plot_curve_data
            per_expert_discrimination_calibration_scores = expert_cal.mse
        else
            discrimination_calibration_curve = missing
            discrimination_calibration_score = missing
            per_expert_discrimination_calibration_curves = missing
            per_expert_discrimination_calibration_scores = missing
            # ...based on ROC curve otherwise
            # optimal_threshold = _get_optimal_threshold_from_ROC(per_class_roc_curves[optimal_threshold_class];
            #                                                     thresholds)
        end

        # # Recalculate `predicted_hard_labels` with this new threshold
        # other_class = optimal_threshold_class == 1 ? 2 : 1
        # for (i, row) in enumerate(eachrow(predicted_soft_labels))
        #     predicted_hard_labels[i] = row[optimal_threshold_class] .>= optimal_threshold ?
        #                                optimal_threshold_class : other_class
        # end
    else
        discrimination_calibration_curve = missing
        discrimination_calibration_score = missing
        per_expert_discrimination_calibration_curves = missing
        per_expert_discrimination_calibration_scores = missing
        optimal_threshold = missing
    end

    # PR curves
    # per_class_pr_curves = [(map(t -> t.true_positive_rate, stats),
    #                         map(t -> t.precision, stats)) for stats in per_class_stats]

    # Stratified kappas
    # stratified_kappas = has_value(strata) ?
    #                     _calculate_stratified_ea_kappas(predicted_hard_labels,
    #                                                     elected_hard_labels, class_count,
    #                                                     strata) : missing

    # Reliability calibration curves
    # per_class_reliability_calibration = map(1:class_count) do class_index
    #     class_probabilities = view(predicted_soft_labels, :, class_index)
    #     return calibration_curve(class_probabilities, elected_hard_labels .== class_index)
    # end
    # per_class_reliability_calibration_curves = map(x -> (mean.(x.bins), x.fractions),
    #                                                per_class_reliability_calibration)
    # per_class_reliability_calibration_scores = map(x -> x.mean_squared_error,
    #                                                per_class_reliability_calibration)

    # Log Spearman correlation, iff this is a binary classification problem
    # if length(classes) == 2 && has_value(votes)
    #     spearman_correlation = _calculate_spearman_correlation(predicted_soft_labels, votes,
    #                                                            classes)
    # else
    #     spearman_correlation = missing
    # end
    return EvaluationRow(; class_labels,
                         confusion_matrix=confusion_matrix(class_count,
                                                           zip(predicted_hard_labels,
                                                               elected_hard_labels)),
                         spearman_correlation, per_class_reliability_calibration_curves,
                         per_class_reliability_calibration_scores,
                         _calculate_ira_kappas(votes, classes)...,
                         _calculate_ea_kappas(predicted_hard_labels, elected_hard_labels,
                                              class_count)..., stratified_kappas,
                         per_class_pr_curves, per_class_roc_curves, per_class_roc_aucs,
                         discrimination_calibration_curve, discrimination_calibration_score,
                         per_expert_discrimination_calibration_curves,
                         per_expert_discrimination_calibration_scores, optimal_threshold,
                         optimal_threshold_class=has_value(optimal_threshold_class) ?
                                                 optimal_threshold_class : missing,
                         thresholds)
end

"""
    evaluation_metrics(args...; optimal_threshold_class=nothing, kwargs...)

Return [`evaluation_metrics_row`](@ref) after converting output `EvaluationRow`
into a `Dict`. For argument details, see [`evaluation_metrics_row`](@ref).
"""
function evaluation_metrics(args...; optimal_threshold_class=nothing, kwargs...)
    row = evaluation_metrics_row(args...;
                                 optimal_threshold_class=something(optimal_threshold_class,
                                                                   missing), kwargs...)
    return _evaluation_row_dict(row)
end

"""
    evaluation_metrics_plot(predicted_hard_labels::AbstractVector,
                            predicted_soft_labels::AbstractMatrix,
                            elected_hard_labels::AbstractVector,
                            classes,
                            thresholds=0.0:0.01:1.0;
                            votes::Union{Nothing,AbstractMatrix}=nothing,
                            strata::Union{Nothing,AbstractVector{Set{T}} where T}=nothing,
                            optimal_threshold_class::Union{Nothing,Integer}=nothing)

Return a plot and dictionary containing a battery of classifier performance
metrics that each compare `predicted_soft_labels` and/or `predicted_hard_labels`
agaist `elected_hard_labels`.

See [`evaluation_metrics`](@ref) for a description of the arguments.

This method is deprecated in favor of calling `evaluation_metrics`
and [`evaluation_metrics_plot`](@ref) separately.
"""
function evaluation_metrics_plot(predicted_hard_labels::AbstractVector,
                                 predicted_soft_labels::AbstractMatrix,
                                 elected_hard_labels::AbstractVector, classes, thresholds;
                                 votes::Union{Nothing,AbstractMatrix}=nothing,
                                 strata::Union{Nothing,AbstractVector{Set{T}} where T}=nothing,
                                 optimal_threshold_class::Union{Nothing,Integer}=nothing)
    Base.depwarn("""
    ```
    evaluation_metrics_plot(predicted_hard_labels::AbstractVector,
                            predicted_soft_labels::AbstractMatrix,
                            elected_hard_labels::AbstractVector, classes, thresholds;
                            votes::Union{Nothing,AbstractMatrix}=nothing,
                            strata::Union{Nothing,AbstractVector{Set{T}} where T}=nothing,
                            optimal_threshold_class::Union{Nothing,Integer}=nothing)
    ```
    has been deprecated in favor of
    ```
    plot_dict = evaluation_metrics(predicted_hard_labels, predicted_soft_labels,
                                   elected_hard_labels, classes, thresholds;
                                   votes, strata, optimal_threshold_class)
    (evaluation_metrics_plot(plot_dict), plot_dict)
    ```
    """, :evaluation_metrics_plot)
    plot_dict = evaluation_metrics(predicted_hard_labels, predicted_soft_labels,
                                   elected_hard_labels, classes, thresholds; votes, strata,
                                   optimal_threshold_class)
    return evaluation_metrics_plot(plot_dict), plot_dict
end

function per_class_confusion_statistics(predicted_soft_labels::AbstractMatrix,
                                        elected_hard_labels::AbstractVector, thresholds)
    class_count = size(predicted_soft_labels, 2)
    return map(1:class_count) do i
        return per_threshold_confusion_statistics(predicted_soft_labels,
                                                  elected_hard_labels,
                                                  thresholds, i)
    end
end

function per_threshold_confusion_statistics(predicted_soft_labels::AbstractMatrix,
                                            elected_hard_labels::AbstractVector, thresholds,
                                            class_index)
    confusions = [confusion_matrix(2) for _ in 1:length(thresholds)]
    for label_index in 1:length(elected_hard_labels)
        predicted_soft_label = predicted_soft_labels[label_index, class_index]
        elected = (elected_hard_labels[label_index] == class_index) + 1
        for (threshold_index, threshold) in enumerate(thresholds)
            predicted = (predicted_soft_label >= threshold) + 1
            confusions[threshold_index][predicted, elected] += 1
        end
    end
    return binary_statistics.(confusions, 2)
end

function per_threshold_confusion_statistics(predicted_soft_labels::AbstractVector,
                                            elected_hard_labels::AbstractVector, thresholds,
                                            class_index)
    confusions = [confusion_matrix(2) for _ in 1:length(thresholds)]
    for label_index in 1:length(elected_hard_labels)
        predicted_soft_label = predicted_soft_labels[label_index]
        elected = (elected_hard_labels[label_index] == class_index) + 1
        for (threshold_index, threshold) in enumerate(thresholds)
            predicted = (predicted_soft_label >= threshold) + 1
            confusions[threshold_index][predicted, elected] += 1
        end
    end
    return binary_statistics.(confusions, 2)
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

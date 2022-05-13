#####
##### confusion matrices
#####

"""
    confusion_matrix(class_count::Integer, hard_label_pairs = ())

Given the iterable `hard_label_pairs` whose `k`th element takes the form
`(first_classifiers_label_for_sample_k, second_classifiers_label_for_sample_k)`,
return the corresponding confusion matrix where `matrix[i, j]` is the number of
samples that the first classifier labeled `i` and the second classifier labeled
`j`.

Note that the returned confusion matrix can be updated in-place with new labels
via `Lighthouse.increment_at!(matrix, more_hard_label_pairs)`.
"""
function confusion_matrix(class_count::Integer, hard_label_pairs=())
    confusion = zeros(Int, class_count, class_count)
    increment_at!(confusion, hard_label_pairs)
    return confusion
end

"""
    accuracy(confusion::AbstractMatrix)

Returns the percentage of matching classifications out of total classifications,
or `NaN` if `all(iszero, confusion)`.

Note that `accuracy(confusion)` is equivalent to overall percent agreement
between `confusion`'s row classifier and column classifier.
"""
function accuracy(confusion::AbstractMatrix)
    total = sum(confusion)
    total == 0 && return NaN
    return tr(confusion) / total
end

"""
    binary_statistics(confusion::AbstractMatrix, class_index)

Treating the rows of `confusion` as corresponding to predicted classifications
and the columns as corresponding to true classifications, return a `NamedTuple`
with the following fields for the given `class_index`:

- `predicted_positives`
- `predicted_negatives`
- `actual_positives`
- `actual_negatives`
- `true_positives`
- `true_negatives`
- `false_positives`
- `false_negatives`
- `true_positive_rate`
- `true_negative_rate`
- `false_positive_rate`
- `false_negative_rate`
- `precision`
- `f1`
"""
function binary_statistics(confusion::AbstractMatrix, class_index::Integer)
    total = sum(confusion)
    predicted_positives = sum(view(confusion, class_index, :))
    predicted_negatives = total - predicted_positives
    actual_positives = sum(view(confusion, :, class_index))
    actual_negatives = total - actual_positives
    true_positives = confusion[class_index, class_index]
    true_negatives = sum(diag(confusion)) - true_positives
    false_positives = predicted_positives - true_positives
    false_negatives = actual_positives - true_positives
    true_positive_rate = (true_positives == 0 && actual_positives == 0) ?
                         (one(true_positives) / one(actual_positives)) :
                         (true_positives / actual_positives)
    true_negative_rate = (true_negatives == 0 && actual_negatives == 0) ?
                         (one(true_negatives) / one(actual_negatives)) :
                         (true_negatives / actual_negatives)
    false_positive_rate = (false_positives == 0 && actual_negatives == 0) ?
                          (zero(false_positives) / one(actual_negatives)) :
                          (false_positives / actual_negatives)
    false_negative_rate = (false_negatives == 0 && actual_positives == 0) ?
                          (zero(false_negatives) / one(actual_positives)) :
                          (false_negatives / actual_positives)
    precision = (true_positives == 0 && predicted_positives == 0) ? NaN :
                (true_positives / predicted_positives)
    f1 = (2 * precision * true_positive_rate) / (precision + true_positive_rate)
    return (; predicted_positives, predicted_negatives, actual_positives, actual_negatives,
            true_positives, true_negatives, false_positives, false_negatives,
            true_positive_rate, true_negative_rate, false_positive_rate,
            false_negative_rate, precision, f1)
end

function binary_statistics(confusion::AbstractMatrix)
    return [binary_statistics(confusion, i) for i in 1:size(confusion, 1)]
end

#####
##### interrater agreement
#####

"""
    cohens_kappa(class_count, hard_label_pairs)

Return `(κ, p₀)` where `κ` is Cohen's kappa and `p₀` percent agreement given
`class_count` and `hard_label_pairs` (these arguments take the same form as
their equivalents in [`confusion_matrix`](@ref)).
"""
function cohens_kappa(class_count, hard_label_pairs)
    all(issubset(pair, 1:class_count) for pair in hard_label_pairs) ||
        throw(ArgumentError("Unexpected class in `hard_label_pairs`."))
    p₀ = accuracy(confusion_matrix(class_count, hard_label_pairs))
    pₑ = _probability_of_chance_agreement(class_count, hard_label_pairs)
    return _cohens_kappa(p₀, pₑ), p₀
end

_cohens_kappa(p₀, pₑ) = (p₀ - pₑ) / (1 - ifelse(pₑ == 1, zero(pₑ), pₑ))

function _probability_of_chance_agreement(class_count, hard_label_pairs)
    labels_1 = (pair[1] for pair in hard_label_pairs)
    labels_2 = (pair[2] for pair in hard_label_pairs)
    x = sum(k -> count(==(k), labels_1) * count(==(k), labels_2), 1:class_count)
    return x / length(hard_label_pairs)^2
end

#####
##### probability distributions
#####

# source: https://github.com/FluxML/Flux.jl/blob/fe85a38d78e225e07a0b75c12b55c8398ae3fe5d/src/layers/stateless.jl#L6
mse(ŷ, y) = sum((ŷ .- y) .^ 2) * 1 // length(y)

"""
    calibration_curve(probabilities, bitmask; bin_count=10)

Given `probabilities` (the predicted probabilities of the positive class) and
`bitmask` (a vector of `Bool`s indicating whether or not the element actually
belonged to the positive class), return `(bins, fractions, totals, mean_squared_error)`
where:

- `bins` a vector with `bin_count` `Pairs` specifying the calibration curve's probability bins
- `fractions`: a vector where `fractions[i]` is the number of values in `probabilities`
  that falls within `bin[i]` over the total number of values within `bin[i]`, or `NaN`
  if the total number of values in `bin[i]` is zero.
- `totals`: a vector where `totals[i]` the total number of values within `bin[i]`.
- `mean_squared_error`: The mean squared error of `fractions` vs. an ideal calibration curve.

This method is similar to the corresponding scikit-learn method:

https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html
"""
function calibration_curve(probabilities, bitmask; bin_count=10)
    bins = probability_bins(bin_count)
    per_bin = [fraction_within(probabilities, bitmask, bin...) for bin in bins]
    fractions, totals = first.(per_bin), last.(per_bin)
    nonempty_indices = findall(!isnan, fractions)
    if !isempty(nonempty_indices)
        ideal = range(mean(first(bins)), mean(last(bins)); length=length(bins))
        mean_squared_error = mse(fractions[nonempty_indices], ideal[nonempty_indices])
    else
        mean_squared_error = NaN
    end
    return (bins=bins, fractions=fractions, totals=totals,
            mean_squared_error=mean_squared_error)
end

function probability_bins(bin_count)
    r = range(0.0, 1.0; length=(bin_count + 1))
    return [begin
                start, stop = r[i], r[i + 1]
                stop = (i + 1) < length(r) ? prevfloat(stop) : stop
                start => stop
            end
            for i in 1:(length(r) - 1)]
end

function fraction_within(values, bitmask, start, stop)
    count = 0
    total = 0
    for (i, value) in enumerate(values)
        if start <= value <= stop
            count += bitmask[i]
            total += 1
        end
    end
    fraction = iszero(total) ? NaN : (count / total)
    return (fraction=fraction, total=total)
end

#####
##### Aggregate metrics to calculate `metrics` schemas defined in `src/row.jl`
#####

"""
    get_tradeoff_metrics(predicted_soft_labels, elected_hard_labels, class_index;
                         thresholds)

Return [`TradeoffMetricsRow`] calculated for the given `class_index`, with the following
fields guaranteed to be non-missing: `roc_curve`, `roc_auc`, pr_curve`,
`reliability_calibration_curve`, `reliability_calibration_score`.`
"""
function get_tradeoff_metrics(predicted_soft_labels, elected_hard_labels, class_index;
                              thresholds)
    stats = per_threshold_confusion_statistics(predicted_soft_labels,
                                               elected_hard_labels, thresholds,
                                               class_index)
    roc_curve = (map(t -> t.false_positive_rate, stats),
                 map(t -> t.true_positive_rate, stats))
    pr_curve = (map(t -> t.true_positive_rate, stats),
                map(t -> t.precision, stats))

    class_probabilities = view(predicted_soft_labels, :, class_index)
    reliability_calibration = calibration_curve(class_probabilities,
                                                elected_hard_labels .== class_index)
    reliability_calibration_curve = (mean.(reliability_calibration.bins),
                                     reliability_calibration.fractions)
    reliability_calibration_score = reliability_calibration.mean_squared_error

    return TradeoffMetricsRow(; class_index, roc_curve,
                              roc_auc=area_under_curve(roc_curve...),
                              pr_curve, reliability_calibration_curve,
                              reliability_calibration_score)
end

"""
    get_tradeoff_metrics_binary_multirater(predicted_soft_labels, elected_hard_labels, class_index;
                                           thresholds)

Return [`TradeoffMetricsRow`] calculated for the given `class_index`. In addition
to metrics calculated by [`get_tradeoff_metrics`](@ref), additionally calculates
`spearman_correlation`-based metrics.
"""
function get_tradeoff_metrics_binary_multirater(predicted_soft_labels, elected_hard_labels,
                                                votes, class_index; thresholds)
    basic_row = get_tradeoff_metrics(predicted_soft_labels, elected_hard_labels,
                                     class_index; thresholds)
    corr = _calculate_spearman_correlation(predicted_soft_labels, votes)
    row = Tables.rowmerge(basic_row,
                          (;
                           spearman_correlation=corr.ρ,
                           spearman_correlation_ci_upper=corr.ci_upper,
                           spearman_correlation_ci_lower=corr.ci_lower,
                           n_samples=corr.n))
    return TradeoffMetricsRow(; row...)
end

"""
    get_hardened_metrics(predicted_hard_labels, elected_hard_labels, class_index;
                         thresholds)

Return [`HardenedMetricsRow`] calculated for the given `class_index`, with the following
field guaranteed to be non-missing: expert-algorithm agreement (`ea_kappa`).
"""
function get_hardened_metrics(predicted_hard_labels, elected_hard_labels, class_index)
    return HardenedMetricsRow(; class_index,
                              ea_kappa=_calculate_ea_kappa(predicted_hard_labels,
                                                           elected_hard_labels,
                                                           class_index))
end

"""
    get_hardened_metrics_multirater(predicted_hard_labels, elected_hard_labels, class_index;
                         thresholds)

Return [`HardenedMetricsRow`] calculated for the given `class_index`. In addition
to metrics calculated by [`get_hardened_metrics`](@ref), additionally calculates
`discrimination_calibration_curve` and `discrimination_calibration_score`.
"""
function get_hardened_metrics_multirater(predicted_hard_labels, elected_hard_labels,
                                         votes, class_index)
    basic_row = get_hardened_metrics(predicted_hard_labels, elected_hard_labels,
                                     class_index)
    cal = _calculate_discrimination_calibration(predicted_hard_labels, votes;
                                                class_of_interest_index=class_index)
    row = Tables.rowmerge(basic_row,
                          (;
                           discrimination_calibration_curve=cal.plot_curve_data,
                           discrimination_calibration_score=cal.mse))
    return HardenedMetricsRow(; row...)
end

"""
    get_hardened_metrics_multiclass(predicted_hard_labels, elected_hard_labels,
                                    class_count)

Return [`HardenedMetricsRow`] calculated over all `class_count` classes. Calculates
expert-algorithm agreement (`ea_kappa`) over all classes, as well as the multiclass
`confusion_matrix`.
"""
function get_hardened_metrics_multiclass(predicted_hard_labels, elected_hard_labels,
                                         class_count)
    ea_kappa = first(cohens_kappa(class_count,
                                  zip(predicted_hard_labels, elected_hard_labels)))
    return HardenedMetricsRow(; class_index=:multiclass,
                              confusion_matrix=confusion_matrix(class_count,
                                                                zip(predicted_hard_labels,
                                                                    elected_hard_labels)),
                              ea_kappa)
end

"""
    get_label_metrics_multirater(votes, class_index)

Return [`LabelMetricsRow`] calculated for the given `class_index`, with the following
field guaranteed to be non-missing: `per_expert_discrimination_calibration_curves`,
`per_expert_discrimination_calibration_scores`, interrater-agreement (`ira_kappa`).
"""
function get_label_metrics_multirater(votes, class_index)
    size(votes, 2) > 1 || throw(ArgumentError("Input `votes` is not multirater (`size(votes) == $(size(votes))`)"))
    expert_cal = _calculate_voter_discrimination_calibration(votes;
                                                             class_of_interest_index=class_index)
    per_expert_discrimination_calibration_curves = expert_cal.plot_curve_data
    per_expert_discrimination_calibration_scores = expert_cal.mse
    return LabelMetricsRow(; class_index, per_expert_discrimination_calibration_curves,
                           per_expert_discrimination_calibration_scores,
                           ira_kappa=_calculate_ira_kappa(votes, class_index))
end

"""
    get_label_metrics_multirater_multiclass(votes, class_count)

Return [`LabelMetricsRow`] calculated over all `class_count` classes. Calculates
the multiclass interrater agreement (`ira_kappa`).
"""
function get_label_metrics_multirater_multiclass(votes, class_count)
    size(votes, 2) > 1 ||
        throw(ArgumentError("Input `votes` is not multirater (`size(votes) == $(size(votes))`)"))
    return LabelMetricsRow(; class_index=:multiclass,
                           ira_kappa=_calculate_ira_kappa_multiclass(votes, class_count))
end

#####
##### Metrics pipelines
#####

"""
    refactored_evaluation_metrics_row(predicted_hard_labels::AbstractVector,
                                      predicted_soft_labels::AbstractMatrix,
                                      elected_hard_labels::AbstractVector, classes;
                                      thresholds=0.0:0.01:1.0,
                                      votes::Union{Nothing,Missing,AbstractMatrix}=nothing,
                                      strata::Union{Nothing,
                                                    AbstractVector{Set{T}} where T}=nothing,
                                      optimal_threshold_class::Union{Missing,Nothing,
                                                                     Integer}=missing)

Drop-in replacement for to-be-deprecated [`evaluation_metrics_row`](@ref), with identical
inputs and outputs.

In service of https://github.com/beacon-biosignals/Lighthouse.jl/pull/69.
"""
function refactored_evaluation_metrics_row(predicted_hard_labels::AbstractVector,
                                           predicted_soft_labels::AbstractMatrix,
                                           elected_hard_labels::AbstractVector, classes,
                                           thresholds=0.0:0.01:1.0;
                                           votes::Union{Nothing,Missing,AbstractMatrix}=nothing,
                                           strata::Union{Nothing,
                                                         AbstractVector{Set{T}} where T}=nothing,
                                           optimal_threshold_class::Union{Missing,Nothing,
                                                                          Integer}=missing)
    class_labels = string.(collect(classes)) # Plots.jl expects this to be an `AbstractVector`
    class_indices = 1:length(classes)

    # Step 1: Calculate all metrics that do not require hardened predictions
    # In our `evaluation_metrics_row` we special-case multirater binary classification,
    # so do that here as well.
    tradeoff_metrics_rows = if length(classes) == 2 && has_value(votes)
        map(ic -> get_tradeoff_metrics_binary_multirater(predicted_soft_labels,
                                                         elected_hard_labels, votes,
                                                         ic;
                                                         thresholds),
            class_indices)
    else
        map(ic -> get_tradeoff_metrics(predicted_soft_labels, elected_hard_labels,
                                       ic;
                                       thresholds),
            class_indices)
    end

    # Step 2a: Choose optimal threshold and use it to harden predictions
    optimal_threshold = missing
    if has_value(optimal_threshold_class) && has_value(votes)
        cal = _calculate_optimal_threshold_from_discrimination_calibration(predicted_soft_labels,
                                                                           votes;
                                                                           thresholds,
                                                                           class_of_interest_index=optimal_threshold_class)
        optimal_threshold = cal.threshold
    elseif has_value(optimal_threshold_class)
        roc_curve = tradeoff_metrics_rows[findfirst(==(optimal_threshold_class),
                                                    tradeoff_metrics_rows.classes),
                                          :]
        optimal_threshold = _get_optimal_threshold_from_ROC(roc_curve, thresholds)
    else
        @warn "Not selecting and/or using optimal threshold; using `predicted_hard_labels` provided by default"
    end

    # Step 2b: Harden predictions with new threshold
    # Note: in new refactored world, should never have hard_predictions before this
    # point, IFF using a threshold to choose a hard label
    if !ismissing(optimal_threshold)
        other_class = optimal_threshold_class == 1 ? 2 : 1
        for (i, row) in enumerate(eachrow(predicted_soft_labels))
            predicted_hard_labels[i] = row[optimal_threshold_class] .>= optimal_threshold ?
                                       optimal_threshold_class : other_class
        end
    end

    # Step 3: Calculate all metrics derived from hardened predictions
    hardened_metrics_table = if has_value(votes)
        map(class_index -> get_hardened_metrics_multirater(predicted_hard_labels,
                                                           elected_hard_labels, votes,
                                                           class_index), class_indices)
    else
        map(class_index -> get_hardened_metrics(predicted_hard_labels,
                                                elected_hard_labels,
                                                class_index), class_indices)
    end
    hardened_metrics_table = vcat(hardened_metrics_table,
                                  get_hardened_metrics_multiclass(predicted_hard_labels,
                                                                  elected_hard_labels,
                                                                  length(classes)))

    # Step 4: Calculate all metrics derived directly from labels (does not depend on
    # predictions)
    labels_metrics_table = LabelMetricsRow[]
    if has_value(votes)
        labels_metrics_table = map(c -> get_label_metrics_multirater(votes, c),
                                   class_indices)
        labels_metrics_table = vcat(labels_metrics_table,
                                    get_label_metrics_multirater_multiclass(votes,
                                                                            length(classes)))
    end

    # Adendum: Not including `stratified_kappas` by default in any of our metrics
    # calculations; including here so as not to fail the deprecation sanity-check
    stratified_kappas = has_value(strata) ?
                        _calculate_stratified_ea_kappas(predicted_hard_labels,
                                                        elected_hard_labels,
                                                        length(classes),
                                                        strata) : missing

    return _evaluation_row(tradeoff_metrics_rows, hardened_metrics_table,
                           labels_metrics_table; optimal_threshold_class, class_labels,
                           thresholds, optimal_threshold, stratified_kappas)
end

function _split_classes_from_multiclass(table)
    table = DataFrame(table; copycols=false)
    nrow(table) == 0 && return (missing, missing)

    # Pull out individual classes
    class_rows = filter(:class_index => c -> isa(c, Int), table)
    sort!(class_rows, :class_index)
    nrow(class_rows) == length(unique(class_rows.class_index)) ||
        throw(ArgumentError("Multiple rows for same class!"))

    # Pull out multiclass
    multi_rows = filter(:class_index => ==(:multiclass), table)
    nrow(multi_rows) > 1 &&
        throw(ArgumentError("More than one `:multiclass` row in table!"))
    multi = nrow(multi_rows) == 1 ? only(multi_rows) : missing
    return class_rows, multi
end

function _values_or_missing(values)
    has_value(values) || return missing
    return all(ismissing, values) ? missing : values
end

_unpack_curves(curve::Union{Missing,Curve}) = ismissing(curve) ? missing : Tuple(curve)
_unpack_curves(curves::AbstractVector{Curve}) = Tuple.(curves)

"""
    _evaluation_row(tradeoff_metrics_table, hardened_metrics_table, label_metrics_table;
                    optimal_threshold_class=missing, class_labels, thresholds,
                    optimal_threshold, stratified_kappas=missing)

Construct an `EvaluationRow` from tables of constituent Metrics schemas, to facilitate
[`refactored_evaluation_metrics_row`](@ref):
- `tradeoff_metrics_table`: table of [`TradeoffMetricsRow`](@ref)s
- `hardened_metrics_table`: table of [`HardenedMetricsRow`](@ref)s
- `label_metrics_table`: table of [`LabelMetricsRow`](@ref)s
"""
function _evaluation_row(tradeoff_metrics_table, hardened_metrics_table,
                         label_metrics_table;
                         optimal_threshold_class=missing, class_labels, thresholds,
                         optimal_threshold, stratified_kappas=missing)
    tradeoff_rows, _ = _split_classes_from_multiclass(tradeoff_metrics_table)
    hardened_rows, hardened_multi = _split_classes_from_multiclass(hardened_metrics_table)
    label_rows, labels_multi = _split_classes_from_multiclass(label_metrics_table)

    # Due to special casing, the following metrics should only be present
    # in the resultant `EvaluationRow` if `optimal_threshold_class` is present
    discrimination_calibration_curve = missing
    discrimination_calibration_score = missing
    per_expert_discrimination_calibration_curves = missing
    per_expert_discrimination_calibration_scores = missing
    if has_value(optimal_threshold_class)
        hardened_row_optimal = only(filter(:class_index => ==(optimal_threshold_class),
                                           hardened_rows))
        discrimination_calibration_curve = hardened_row_optimal.discrimination_calibration_curve
        discrimination_calibration_score = hardened_row_optimal.discrimination_calibration_score

        label_row_optimal = only(filter(:class_index => ==(optimal_threshold_class),
                                        label_rows))
        per_expert_discrimination_calibration_curves = label_row_optimal.per_expert_discrimination_calibration_curves
        per_expert_discrimination_calibration_scores = label_row_optimal.per_expert_discrimination_calibration_scores
    end

    # Similarly, due to separate special casing, only get the spearman correlation coefficient
    # from a binary classification problem. It is calculated for both classes, but is
    # identical, so grab it from the first
    spearman_correlation = missing
    if length(class_labels) == 2
        row = first(tradeoff_rows)
        spearman_correlation = ismissing(row.spearman_correlation) ? missing :
                               (; ρ=row.spearman_correlation, n=row.n_samples,
                                ci_lower=row.spearman_correlation_ci_lower,
                                ci_upper=row.spearman_correlation_ci_upper)
    end
    return EvaluationRow(;
                         # ...from hardened_metrics_table
                         confusion_matrix=_values_or_missing(hardened_multi.confusion_matrix),
                         multiclass_kappa=_values_or_missing(hardened_multi.ea_kappa),
                         per_class_kappas=_values_or_missing(hardened_rows.ea_kappa),
                         discrimination_calibration_curve=_unpack_curves(discrimination_calibration_curve),
                         discrimination_calibration_score,

                         # ...from tradeoff_metrics_table
                         per_class_roc_curves=_unpack_curves(_values_or_missing(tradeoff_rows.roc_curve)),
                         per_class_roc_aucs=_values_or_missing(tradeoff_rows.roc_auc),
                         per_class_pr_curves=_unpack_curves(_values_or_missing(tradeoff_rows.pr_curve)),
                         spearman_correlation,
                         per_class_reliability_calibration_curves=_unpack_curves(_values_or_missing(tradeoff_rows.reliability_calibration_curve)),
                         per_class_reliability_calibration_scores=_values_or_missing(tradeoff_rows.reliability_calibration_score),

                         # from label_metrics_table
                         per_expert_discrimination_calibration_curves=_unpack_curves(_values_or_missing(per_expert_discrimination_calibration_curves)),
                         multiclass_IRA_kappas=_values_or_missing(labels_multi.ira_kappa),
                         per_class_IRA_kappas=_values_or_missing(label_rows.ira_kappa),
                         per_expert_discrimination_calibration_scores,

                         # from kwargs:
                         optimal_threshold_class, class_labels, thresholds,
                         optimal_threshold, stratified_kappas)
end

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
or `missing` if `all(iszero, confusion)`.

Note that `accuracy(confusion)` is equivalent to overall percent agreement
between `confusion`'s row classifier and column classifier.
"""
function accuracy(confusion::AbstractMatrix)
    total = sum(confusion)
    total == 0 && return missing
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
    precision = (true_positives == 0 && predicted_positives == 0) ? missing :
                (true_positives / predicted_positives)
    return (predicted_positives=predicted_positives,
            predicted_negatives=predicted_negatives, actual_positives=actual_positives,
            actual_negatives=actual_negatives, true_positives=true_positives,
            true_negatives=true_negatives, false_positives=false_positives,
            false_negatives=false_negatives, true_positive_rate=true_positive_rate,
            true_negative_rate=true_negative_rate, false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate, precision=precision)
end

function gain(actual_positives, actual_negatives)
    π = actual_positives / (actual_positives + actual_negatives)
    return π / (1 - π)
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
    @assert all(issubset(pair, 1:class_count) for pair in hard_label_pairs)
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
  that falls within `bin[i]` over the total number of values within `bin[i]`, or `missing`
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
    nonempty_indices = findall(!ismissing, fractions)
    if !isempty(nonempty_indices)
        ideal = range(mean(first(bins)), mean(last(bins)); length=length(bins))
        mean_squared_error = mse(fractions[nonempty_indices], ideal[nonempty_indices])
    else
        mean_squared_error = missing
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
            end for i in 1:(length(r) - 1)]
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
    fraction = iszero(total) ? missing : (count / total)
    return (fraction=fraction, total=total)
end

#####
##### `AbstractClassifier` Interface
#####
# This section contains all the functions that new `AbstractClassifier` subtypes
# should overload in order to utilize common Lighthouse functionality

"""
    AbstractClassifier

An abstract type whose subtypes `C<:AbstractClassifier` must implement:

- [`Lighthouse.classes`](@ref)
- [`Lighthouse.train!`](@ref)
- [`Lighthouse.loss_and_prediction`](@ref)

Subtypes may additionally overload default implementations for:

- [`Lighthouse.onehot`](@ref)
- [`Lighthouse.onecold`](@ref)
- [`Lighthouse.is_early_stopping_exception`](@ref)

The `AbstractClassifier` interface is built upon the expectation that any
multiclass label will be represented in one of two standardized forms:

- "soft label": a probability distribution vector where the `i`th element is the
  probability assigned to the `i`th class in `classes(classifier)`.
- "hard label": the interger index of a corresponding class in `classes(classifier)`.

Internally, Lighthouse converts hard labels to soft labels via `onehot` and soft
labels to hard labels via `onecold`.

See also: [`learn!`](@ref)
"""
abstract type AbstractClassifier end

"""
    Lighthouse.classes(classifier::AbstractClassifier)

Return a `Vector` or `Tuple` of class values for `classifier`.

This method must be implemented for each `AbstractClassifier` subtype.
"""
function classes end

"""
    Lighthouse.train!(classifier::AbstractClassifier, batches, logger)

Train `classifier` on the iterable `batches` for a single epoch. This function
is called once per epoch by [`learn!`](@ref).

This method must be implemented for each `AbstractClassifier` subtype. Implementers
should ensure that the training loss is properly logged to `logger` by calling
`Lighthouse.log_value!(logger, "train/loss_per_batch", batch_loss)` for
each batch in `batches`.
"""
function train! end

"""
    Lighthouse.loss_and_prediction(classifier::AbstractClassifier,
                                   input_batch::AbstractArray,
                                   args...)

Return `(loss, soft_label_batch)` given `input_batch` and any additional `args`
provided by the caller; `loss` is a scalar, which `soft_label_batch` is a matrix
with `length(classes(classifier))` rows and `size(input_batch)`.

Specifically, the `i`th column of `soft_label_batch` is `classifier`'s soft
label prediction for the `i`th sample in `input_batch`.

This method must be implemented for each `AbstractClassifier` subtype.
"""
function loss_and_prediction end

"""
    Lighthouse.onehot(classifier::AbstractClassifier, hard_label)

Return the one-hot encoded probability distribution vector corresponding to the
given `hard_label`. `hard_label` must be an integer index in the range
`1:length(classes(classifier))`.
"""
function onehot(classifier::AbstractClassifier, hard_label)
    result = fill(false, length(classes(classifier)))
    result[hard_label] = true
    return result
end

"""
    Lighthouse.onecold(classifier::AbstractClassifier, soft_label)

Return the hard label (integer index in the range `1:length(classes(classifier))`)
corresponding to the given `soft_label` (one-hot encoded probability distribution vector).

By default, this function returns `argmax(soft_label)`.
"""
onecold(classifier::AbstractClassifier, soft_label) = argmax(soft_label)

"""
    Lighthouse.is_early_stopping_exception(classifier::AbstractClassifier, exception)

Return `true` if `exception` should be considered an "early-stopping exception"
(e.g. `Flux.Optimise.StopException`), rather than rethrown from [`learn!`](@ref).

This function returns `false` by default, but can be overloaded by subtypes of
`AbstractClassifier` that employ exceptions as early-stopping mechanisms.
"""
is_early_stopping_exception(::AbstractClassifier, ::Any) = false

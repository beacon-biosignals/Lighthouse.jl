
#####
##### `EvaluationRow`
#####

# Arrow can't handle matrices---so when we write/read matrices, we have to pack and unpack them o_O
# https://github.com/apache/arrow-julia/issues/125
vec_to_mat(mat::AbstractMatrix) = mat

function vec_to_mat(vec::AbstractVector)
    n = isqrt(length(vec))
    return reshape(vec, n, n)
end

vec_to_mat(x::Missing) = return missing

# Redefinition is workaround for https://github.com/beacon-biosignals/Legolas.jl/issues/9
const EVALUATION_ROW_SCHEMA = Legolas.Schema("lighthouse.evaluation@1")

"""
    const EvaluationRow = Legolas.@row("lighthouse.evaluation@1",
                                   class_labels::Union{Missing,Vector{String}},
                                   confusion_matrix::Union{Missing,Array{Int64}} = vec_to_mat(confusion_matrix),
                                   discrimination_calibration_curve::Union{Missing,
                                                                           Tuple{Vector{Float64},
                                                                                 Vector{Float64}}},
                                   discrimination_calibration_score::Union{Missing,Float64},
                                   multiclass_IRA_kappas::Union{Missing,Float64},
                                   multiclass_kappa::Union{Missing,Float64},
                                   optimal_threshold::Union{Missing,Float64},
                                   optimal_threshold_class::Union{Missing,Int64},
                                   per_class_IRA_kappas::Union{Missing,Vector{Float64}},
                                   per_class_kappas::Union{Missing,Vector{Float64}},
                                   stratified_kappas::Union{Missing,
                                                            Vector{NamedTuple{(:per_class,
                                                                               :multiclass,
                                                                               :n),
                                                                              Tuple{Vector{Float64},
                                                                                    Float64,
                                                                                    Int64}}}},
                                   per_class_pr_curves::Union{Missing,
                                                              Vector{Tuple{Vector{Float64},
                                                                           Vector{Float64}}}},
                                   per_class_reliability_calibration_curves::Union{Missing,
                                                                                   Vector{Tuple{Vector{Float64},
                                                                                                Vector{Float64}}}},
                                   per_class_reliability_calibration_scores::Union{Missing,
                                                                                   Vector{Float64}},
                                   per_class_roc_aucs::Union{Missing,Vector{Float64}},
                                   per_class_roc_curves::Union{Missing,
                                                               Vector{Tuple{Vector{Float64},
                                                                            Vector{Float64}}}},
                                   per_expert_discrimination_calibration_curves::Union{Missing,
                                                                                       Vector{Tuple{Vector{Float64},
                                                                                                    Vector{Float64}}}},
                                   per_expert_discrimination_calibration_scores::Union{Missing,
                                                                                       Vector{Float64}},
                                   spearman_correlation::Union{Missing,
                                                               NamedTuple{(:ρ, :n,
                                                                           :ci_lower,
                                                                           :ci_upper),
                                                                          Tuple{Float64,
                                                                                Int64,
                                                                                Float64,
                                                                                Float64}}},
                                   thresholds::Union{Missing,Vector{Float64}})
    EvaluationRow(evaluation_row_dict::Dict{String, Any}) -> EvaluationRow

A type alias for [`Legolas.Row{typeof(Legolas.Schema("lighthouse.evaluation@1"))}`](https://beacon-biosignals.github.io/Legolas.jl/stable/#Legolas.@row)
representing the output metrics computed by [`evaluation_metrics_row`](@ref) and
[`evaluation_metrics`](@ref).

Constructor that takes `evaluation_row_dict` converts [`evaluation_metrics`](@ref)
`Dict` of metrics results (e.g. from Lighthouse <v0.14.0) into an [`EvaluationRow`](@ref).
"""
const EvaluationRow = Legolas.@row("lighthouse.evaluation@1",
                                   class_labels::Union{Missing,Vector{String}},
                                   confusion_matrix::Union{Missing,Array{Int64}} = vec_to_mat(confusion_matrix),
                                   discrimination_calibration_curve::Union{Missing,
                                                                           Tuple{Vector{Float64},
                                                                                 Vector{Float64}}},
                                   discrimination_calibration_score::Union{Missing,Float64},
                                   multiclass_IRA_kappas::Union{Missing,Float64},
                                   multiclass_kappa::Union{Missing,Float64},
                                   optimal_threshold::Union{Missing,Float64},
                                   optimal_threshold_class::Union{Missing,Int64},
                                   per_class_IRA_kappas::Union{Missing,Vector{Float64}},
                                   per_class_kappas::Union{Missing,Vector{Float64}},
                                   stratified_kappas::Union{Missing,
                                                            Vector{NamedTuple{(:per_class,
                                                                               :multiclass,
                                                                               :n),
                                                                              Tuple{Vector{Float64},
                                                                                    Float64,
                                                                                    Int64}}}},
                                   per_class_pr_curves::Union{Missing,
                                                              Vector{Tuple{Vector{Float64},
                                                                           Vector{Float64}}}},
                                   per_class_reliability_calibration_curves::Union{Missing,
                                                                                   Vector{Tuple{Vector{Float64},
                                                                                                Vector{Float64}}}},
                                   per_class_reliability_calibration_scores::Union{Missing,
                                                                                   Vector{Float64}},
                                   per_class_roc_aucs::Union{Missing,Vector{Float64}},
                                   per_class_roc_curves::Union{Missing,
                                                               Vector{Tuple{Vector{Float64},
                                                                            Vector{Float64}}}},
                                   per_expert_discrimination_calibration_curves::Union{Missing,
                                                                                       Vector{Tuple{Vector{Float64},
                                                                                                    Vector{Float64}}}},
                                   per_expert_discrimination_calibration_scores::Union{Missing,
                                                                                       Vector{Float64}},
                                   spearman_correlation::Union{Missing,
                                                               NamedTuple{(:ρ, :n,
                                                                           :ci_lower,
                                                                           :ci_upper),
                                                                          Tuple{Float64,
                                                                                Int64,
                                                                                Float64,
                                                                                Float64}}},
                                   thresholds::Union{Missing,Vector{Float64}})

function Legolas.Row{S}(evaluation_row_dict::Dict) where {S<:Legolas.Schema{Symbol("lighthouse.evaluation"),
                                                                            1}}
    row = (; (Symbol(k) => v for (k, v) in pairs(evaluation_row_dict))...)
    return EvaluationRow(row)
end

"""
    _evaluation_row_dict(row::EvaluationRow) -> Dict{String,Any}

Convert [`EvaluationRow`](@ref) into `::Dict{String, Any}` results, as are
output by `[`evaluation_metrics`](@ref)` (and predated use of `EvaluationRow` in
Lighthouse <v0.14.0).
"""
function _evaluation_row_dict(row::EvaluationRow)
    return Dict(string(k) => v for (k, v) in pairs(NamedTuple(row)) if !ismissing(v))
end

#####
##### `ObservationRow`
#####

# Redefinition is workaround for https://github.com/beacon-biosignals/Legolas.jl/issues/9
const OBSERVATION_ROW_SCHEMA = Legolas.Schema("lighthouse.observation@1")
"""
    const ObservationRow = Legolas.@row("lighthouse.observation@1",
                                        predicted_hard_label::Int64,
                                        predicted_soft_labels::Vector{Float32},
                                        elected_hard_label::Int64,
                                        votes::Union{Missing,Vector{Int64}})

A type alias for [`Legolas.Row{typeof(Legolas.Schema("lighthouse.observation@1"))}`](https://beacon-biosignals.github.io/Legolas.jl/stable/#Legolas.@row)
representing the per-observation input values required to compute [`evaluation_metrics_row`](@ref).
"""
const ObservationRow = Legolas.@row("lighthouse.observation@1",
                                    predicted_hard_label::Int64,
                                    predicted_soft_labels::Vector{Float32},
                                    elected_hard_label::Int64,
                                    votes::Union{Missing,Vector{Int64}})

function _observation_table_to_inputs(observation_table)
    Legolas.validate(observation_table, OBSERVATION_ROW_SCHEMA)
    df_table = Tables.columns(observation_table)
    votes = missing
    if any(ismissing, df_table.votes) && !all(ismissing, df_table.votes)
        throw(ArgumentError("`:votes` must either be all `missing` or contain no `missing`"))
    end
    votes = any(ismissing, df_table.votes) ? missing :
            transpose(reduce(hcat, df_table.votes))

    predicted_soft_labels = transpose(reduce(hcat, df_table.predicted_soft_labels))
    return (; predicted_hard_labels=df_table.predicted_hard_label, predicted_soft_labels,
            elected_hard_labels=df_table.elected_hard_label, votes)
end

function _inputs_to_observation_table(; predicted_hard_labels::AbstractVector,
                                      predicted_soft_labels::AbstractMatrix,
                                      elected_hard_labels::AbstractVector,
                                      votes::Union{Nothing,Missing,AbstractMatrix}=nothing)
    votes_itr = has_value(votes) ? eachrow(votes) :
                (missing for _ in 1:length(predicted_hard_labels))
    predicted_soft_labels_itr = eachrow(predicted_soft_labels)
    if !(length(predicted_hard_labels) == length(predicted_soft_labels_itr) ==
         length(elected_hard_labels) == length(votes_itr))
        throw(DimensionMismatch("Inputs do not all have the same number of observations"))
    end
    observation_table = map(predicted_hard_labels, elected_hard_labels,
                            predicted_soft_labels_itr,
                            votes_itr) do predicted_hard_label, elected_hard_label,
                                          predicted_soft_labels, votes
        return ObservationRow(; predicted_hard_label, elected_hard_label,
                              predicted_soft_labels, votes)
    end
    Legolas.validate(observation_table, OBSERVATION_ROW_SCHEMA)
    return observation_table
end

#####
##### Metrics rows
#####

"""
    ClassRow = Legolas.@row("lighthouse.class@1", class::Union{Int,Symbol})

A type alias for [`Legolas.Row{typeof(Legolas.Schema("lighthouse.class@1"))}`](https://beacon-biosignals.github.io/Legolas.jl/stable/#Legolas.@row)
representing a single column that either repsresnts a single class or `:multiclass`.
"""
const ClassRow = Legolas.@row("lighthouse.class@1", class_index::Union{Int64,Symbol})

"""
    LabelMetricsRow = Legolas.@row("lighthouse.label-metrics@1" > "lighthouse.class@1",
                                     IRA_kappa::Union{Missing,Float64},
                                     per_expert_discrimination_calibration_curves::Union{Missing,
                                                                                         Vector{Tuple{Vector{Float64},
                                                                                                      Vector{Float64}}}},
                                     per_expert_discrimination_calibration_scores::Union{Missing,
                                                                                         Vector{Float64}})

A type alias for [`Legolas.Row{typeof(Legolas.Schema("label-metrics@1"))}`](https://beacon-biosignals.github.io/Legolas.jl/stable/#Legolas.@row)
representing metrics calculated over labels provided by multiple labelers.
See also [`get_label_metrics`](@ref).
"""
const LabelMetricsRow = Legolas.@row("lighthouse.label-metrics@1" > "lighthouse.class@1",
                                     ira_kappa::Union{Missing,Float64},
                                     per_expert_discrimination_calibration_curves::Union{Missing,
                                                                                         Vector{Tuple{Vector{Float64},
                                                                                                      Vector{Float64}}}},
                                     per_expert_discrimination_calibration_scores::Union{Missing,
                                                                                         Vector{Float64}})

"""
    HardenedMetricsRow = Legolas.@row("lighthouse.hardened-metrics@1" >
                                        "lighthouse.class@1",
                                        confusion_matrix::Union{Missing,Array{Int64}} = vec_to_mat(confusion_matrix),
                                        discrimination_calibration_curve::Union{Missing,
                                                                                Tuple{Vector{Float64},
                                                                                      Vector{Float64}}},
                                        discrimination_calibration_score::Union{Missing,
                                                                                Float64},
                                        kappa::Union{Missing,Float64})

A type alias for [`Legolas.Row{typeof(Legolas.Schema("hardened-metrics@1"))}`](https://beacon-biosignals.github.io/Legolas.jl/stable/#Legolas.@row)
representing metrics calculated over predicted hard labels.
See also [`get_hardened_metrics`](@ref).
"""
const HardenedMetricsRow = Legolas.@row("lighthouse.hardened-metrics@1" >
                                        "lighthouse.class@1",
                                        confusion_matrix::Union{Missing,Array{Int64}} = vec_to_mat(confusion_matrix),
                                        discrimination_calibration_curve::Union{Missing,
                                                                                Tuple{Vector{Float64},
                                                                                      Vector{Float64}}},
                                        discrimination_calibration_score::Union{Missing,
                                                                                Float64},
                                        ea_kappa::Union{Missing,Float64})

"""
    TradeoffMetricsRow = Legolas.@row("lighthouse.tradeoff-metrics@1" >
                                        "lighthouse.class@1",
                                        pr_curve::Union{Missing,
                                                        Tuple{Vector{Float64},
                                                              Vector{Float64}}},
                                        roc_auc::Union{Missing,Float64},
                                        roc_curve::Union{Missing,
                                                         Tuple{Vector{Float64},
                                                               Vector{Float64}}},
                                        spearman_correlation::Union{Missing,
                                                                    NamedTuple{(:ρ, :n,
                                                                                :ci_lower,
                                                                                :ci_upper),
                                                                               Tuple{Float64,
                                                                                     Int64,
                                                                                     Float64,
                                                                                     Float64}}},
                                        reliability_calibration_curve::Union{Missing,
                                                                             Tuple{Vector{Float64},
                                                                                   Vector{Float64}}},
                                        reliability_calibration_score::Union{Missing,
                                                                             Float64})

A type alias for [`Legolas.Row{typeof(Legolas.Schema("tradeoff-metrics@1"))}`](https://beacon-biosignals.github.io/Legolas.jl/stable/#Legolas.@row)
representing metrics calculated over predicted soft labels.
See also [`get_tradeoff_metrics`](@ref).
"""
const TradeoffMetricsRow = Legolas.@row("lighthouse.tradeoff-metrics@1" >
                                        "lighthouse.class@1",
                                        roc_curve::Tuple{Vector{Float64},Vector{Float64}},
                                        roc_auc::Float64,
                                        pr_curve::Tuple{Vector{Float64},Vector{Float64}},
                                        spearman_correlation::Union{Missing,
                                                                    NamedTuple{(:ρ, :n,
                                                                                :ci_lower,
                                                                                :ci_upper),
                                                                               Tuple{Float64,
                                                                                     Int64,
                                                                                     Float64,
                                                                                     Float64}}},
                                        reliability_calibration_curve::Union{Missing,
                                                                             Tuple{Vector{Float64},
                                                                                   Vector{Float64}}},
                                        reliability_calibration_score::Union{Missing,
                                                                             Float64})

function _split_classes_from_multiclass(table)
    table = DataFrame(table; copycols=false)

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

# Helper constructor method to help sanity-check refactor
function Legolas.Row{S}(tradeoff_metrics_table, hardened_metrics_table, label_metrics_table;
                        optimal_threshold_class=missing, class_labels, thresholds,
                        optimal_threshold,
                        stratified_kappas=missing) where {S<:Legolas.Schema{Symbol("lighthouse.evaluation"),
                                                                            1}}
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
        spearman_correlation = first(tradeoff_rows).spearman_correlation
    end
    return EvaluationRow(;
                         # ...from hardened_metrics_table
                         confusion_matrix=_values_or_missing(hardened_multi.confusion_matrix),
                         multiclass_kappa=_values_or_missing(hardened_multi.ea_kappa),
                         per_class_kappas=_values_or_missing(hardened_rows.ea_kappa),
                         discrimination_calibration_curve,
                         discrimination_calibration_score,

                         # ...from tradeoff_metrics_table
                         per_class_roc_curves=_values_or_missing(tradeoff_rows.roc_curve),
                         per_class_roc_aucs=_values_or_missing(tradeoff_rows.roc_auc),
                         per_class_pr_curves=_values_or_missing(tradeoff_rows.pr_curve),
                         spearman_correlation,
                         per_class_reliability_calibration_curves=_values_or_missing(tradeoff_rows.reliability_calibration_curve),
                         per_class_reliability_calibration_scores=_values_or_missing(tradeoff_rows.reliability_calibration_score),

                         # from label_metrics_table
                         per_expert_discrimination_calibration_curves=_values_or_missing(per_expert_discrimination_calibration_curves),
                         multiclass_IRA_kappas=_values_or_missing(labels_multi.ira_kappa),
                         per_class_IRA_kappas=_values_or_missing(label_rows.ira_kappa),
                         per_expert_discrimination_calibration_scores,

                         # from kwargs:
                         optimal_threshold_class, class_labels, thresholds,
                         optimal_threshold, stratified_kappas)
end



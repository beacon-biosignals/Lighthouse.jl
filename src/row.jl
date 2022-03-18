# Arrow can't handle matrices---so when we write/read matrices, we have to pack and unpack them o_O
# https://github.com/apache/arrow-julia/issues/125
vec_to_mat(mat::AbstractMatrix) = mat

function vec_to_mat(vec::AbstractVector)
    n = isqrt(length(vec))
    return reshape(vec, n, n)
end

vec_to_mat(x::Missing) = return missing

#####
##### `EvaluationRow`
#####

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

A type alias for [`Legolas.Row{typeof(Legolas.Schema("lighthouse.evaluation@1@1"))}`](https://beacon-biosignals.github.io/Legolas.jl/stable/#Legolas.@row)
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
##### `LabeledInputRow`
#####

# Redefinition is workaround for https://github.com/beacon-biosignals/Legolas.jl/issues/9
const OBSERVATION_ROW_SCHEMA = Legolas.Schema("lighthouse.observation@1")
"""
    const ObservationRow = Legolas.@row("lighthouse.observation@1",
                                    predicted_hard_labels::Int64,
                                    predicted_soft_labels::Vector{Float32},
                                    elected_hard_labels::Int64,
                                    votes::Union{Missing,Vector{Int64}})

A type alias for [`Legolas.Row{typeof(Legolas.Schema("lighthouse.observation@1"))}`](https://beacon-biosignals.github.io/Legolas.jl/stable/#Legolas.@row)
representing the per-observation input values required to compute [`evaluation_metrics_row`](@ref).
"""
const ObservationRow = Legolas.@row("lighthouse.observation@1",
                                    predicted_hard_labels::Int64,
                                    predicted_soft_labels::Vector{Float32},
                                    elected_hard_labels::Int64,
                                    votes::Union{Missing,Vector{Int64}})

function _obervation_table_to_inputs(observation_table)
    Legolas.validate(observation_table, OBSERVATION_ROW_SCHEMA)
    df_table = DataFrame(observation_table; copycols=false)
    votes = missing
    if any(ismissing, df_table.votes) && !all(ismissing, df_table.votes)
        throw(ArgumentError("`:votes` must either be all `missing` or contain no `missing`"))
    end
    votes = any(ismissing, df_table.votes) ? missing :
            reduce(hcat, df_table.votes)'
    predicted_soft_labels = reduce(hcat, df_table.predicted_soft_labels)'
    return (; df_table.predicted_hard_labels, predicted_soft_labels,
            df_table.elected_hard_labels, votes)
end

function _inputs_to_obervation_table(; predicted_hard_labels::AbstractVector,
                                     predicted_soft_labels::AbstractMatrix,
                                     elected_hard_labels::AbstractVector,
                                     votes::Union{Nothing,Missing,AbstractMatrix}=nothing)
    votes = (isnothing(votes) || ismissing(votes)) ? missing : collect(eachrow(votes))
    observations = DataFrame(; predicted_hard_labels, elected_hard_labels,
                             predicted_soft_labels=collect(eachrow(predicted_soft_labels)),
                             votes)
    observation_table = DataFrame([ObservationRow(r) for r in eachrow(observations)])
    Legolas.validate(observation_table, OBSERVATION_ROW_SCHEMA)
    return observation_table
end
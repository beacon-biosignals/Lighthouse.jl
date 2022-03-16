function vec_to_mat(mat::AbstractMatrix)
    @info "here"
    return mat
end
function vec_to_mat(vec::AbstractVector)
    @info "ok..."
    n = isqrt(length(vec))
    return reshape(vec, n, n)
end
vec_to_mat(x::Missing) = (@info "why"; @info typeof(x); return missing)

# Redefinition is workaround for https://github.com/beacon-biosignals/Legolas.jl/issues/9
const EVALUATION_ROW_SCHEMA = Legolas.Schema("lighthouse.evaluation@1")
const EvaluationRow = Legolas.@row("lighthouse.evaluation@1",
                                   class_labels::Union{Missing,Vector{String}},
                                   confusion_matrix::Union{Missing,Matrix{Int64}} = vec_to_mat(confusion_matrix),
                                   discrimination_calibration_curve::Union{Missing,
                                                                           Tuple{Vector{Float64},
                                                                                 Vector{Union{Missing,
                                                                                              Float64}}}},
                                   discrimination_calibration_score::Union{Missing,Float64},
                                   multiclass_IRA_kappas::Union{Missing,Float64},
                                   multiclass_kappa::Union{Missing,Float64},
                                   optimal_threshold::Union{Missing,Float64},
                                   optimal_threshold_class::Union{Missing,Int64},
                                   per_class_kappas::Union{Missing,Vector{Float64}},
                                   per_class_pr_curves::Union{Missing,
                                                              Vector{Tuple{Vector{Float64},
                                                                           Vector{Union{Missing,
                                                                                        Float64}}}}},
                                   per_class_reliability_calibration_curves::Union{Missing,
                                                                                   Vector{Tuple{Vector{Float64},
                                                                                                Vector{Union{Missing,
                                                                                                             Float64}}}}},
                                   per_class_reliability_calibration_scores::Union{Missing,
                                                                                   Vector{Float64}},
                                   per_class_roc_aucs::Union{Missing,Vector{Float64}},
                                   per_class_roc_curves::Union{Missing,
                                                               Vector{Tuple{Vector{Float64},
                                                                            Vector{Float64}}}},
                                   per_expert_discrimination_calibration_curves::Union{Missing,
                                                                                       Vector{Tuple{Vector{Float64},
                                                                                                    Vector{Union{Missing,
                                                                                                                 Float64}}}}},
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

"""
    EvaluationRow(evaluation_row_dict::Dict{String, Any}) -> EvaluationRow

Convert `Dict` of  [`evaluation_metrics`](@ref) results (e.g. from Lighthouse <v0.14.0)
into an [`EvaluationRow`](@ref).
"""
function EvaluationRow(evaluation_row_dict::Dict{String,Any})
    @info "this"
    row = (; Dict(Symbol(k) => v for (k, v) in pairs(evaluation_row_dict))...)
    return EvaluationRow(row)
end


"""
    _evaluation_row_dict(row::EvaluationRow) -> Dict{String,Any}

Convert [`EvaluationRow`](@ref) into `::Dict{String, Any}` results, as are
output by `[`evaluation_metrics`](@ref)` (and predated use of `EvaluationRow` in
Lighthouse <v0.14.0.
"""
function _evaluation_row_dict(row::EvaluationRow)
    return Dict(string(k) => v for (k, v) in pairs(NamedTuple(row)) if !ismissing(v))
end
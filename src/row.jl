vec_to_mat(mat::AbstractMatrix) = mat
function vec_to_mat(vec::AbstractVector)
    n = isqrt(length(vec))
    return reshape(vec, n, n)
end
vec_to_mat(::Missing) = missing

const EvaluationRow = Legolas.@row("lighthouse.evaluation@1",
    class_labels::Union{Missing, Vector{String}},
    confusion_matrix::Union{Missing, Matrix{Int64}}=vec_to_mat(confusion_matrix),
    discrimination_calibration_curve::Union{Missing, Tuple{Vector{Float64}, Vector{Union{Missing, Float64}}}},
    discrimination_calibration_score::Union{Missing, Float64},
    multiclass_IRA_kappas::Union{Missing, Float64}, 
    multiclass_kappa::Union{Missing, Float64}, 
    optimal_threshold::Union{Missing, Float64},
    per_class_kappas::Union{Missing, Vector{Float64}},   
    per_class_pr_curves::Union{Missing, Vector{Tuple{Vector{Float64}, Vector{Union{Missing, Float64}}}}},
    per_class_reliability_calibration_curves::Union{Missing, Vector{Tuple{Vector{Float64}, Vector{Union{Missing, Float64}}}}},
    per_class_reliability_calibration_scores::Union{Missing, Vector{Float64}},
    per_class_roc_aucs::Union{Missing, Vector{Float64}}, 
    per_class_roc_curves::Union{Missing, Vector{Tuple{Vector{Float64}, Vector{Float64}}}},
    per_expert_discrimination_calibration_curves::Union{Missing, Vector{Tuple{Vector{Float64}, Vector{Union{Missing, Float64}}}}},
    per_expert_discrimination_calibration_scores::Union{Missing, Vector{Float64}},
    spearman_correlation::Union{Missing, NamedTuple{(:ρ, :n, :ci_lower, :ci_upper), Tuple{Float64, Int64, Float64, Float64}}}, 
    thresholds::Union{Missing, Vector{Float64}})

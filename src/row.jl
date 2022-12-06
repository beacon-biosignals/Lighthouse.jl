
#####
##### `EvaluationObject
#####

# Arrow can't handle matrices---so when we write/read matrices, we have to pack and unpack them o_O
# https://github.com/apache/arrow-julia/issues/125
vec_to_mat(mat::AbstractMatrix) = mat
function vec_to_mat(vec::AbstractVector)
    n = isqrt(length(vec))
    return reshape(vec, n, n)
end
vec_to_mat(x::Missing) = return missing

const GenericCurve = Tuple{Vector{Float64},Vector{Float64}}
@schema "lighthouse.evaluation" Evaluation
@version EvaluationV1 begin
    class_labels::Union{Missing,Vector{String}}
    confusion_matrix::Union{Missing,Array{Int64}} = vec_to_mat(confusion_matrix)
    discrimination_calibration_curve::Union{Missing,GenericCurve}
    discrimination_calibration_score::Union{Missing,Float64}
    multiclass_IRA_kappas::Union{Missing,Float64}
    multiclass_kappa::Union{Missing,Float64}
    optimal_threshold::Union{Missing,Float64}
    optimal_threshold_class::Union{Missing,Int64}
    per_class_IRA_kappas::Union{Missing,Vector{Float64}}
    per_class_kappas::Union{Missing,Vector{Float64}}
    stratified_kappas::Union{Missing,
                             Vector{@NamedTuple{per_class::Vector{Float64},
                                                multiclass::Float64,
                                                n::Int64}}}
    per_class_pr_curves::Union{Missing,Vector{GenericCurve}}
    per_class_reliability_calibration_curves::Union{Missing,Vector{GenericCurve}}
    per_class_reliability_calibration_scores::Union{Missing,Vector{Float64}}
    per_class_roc_aucs::Union{Missing,Vector{Float64}}
    per_class_roc_curves::Union{Missing,Vector{GenericCurve}}
    per_expert_discrimination_calibration_curves::Union{Missing,Vector{GenericCurve}}
    per_expert_discrimination_calibration_scores::Union{Missing,Vector{Float64}}
    spearman_correlation::Union{Missing,
                                @NamedTuple{p::Float64,
                                            n::Int64,
                                            ci_lower::Float64,
                                            ci_upper::Float64}}
    thresholds::Union{Missing,Vector{Float64}}
end
"""
    @version EvaluationV1 begin
        class_labels::Union{Missing,Vector{String}}
        confusion_matrix::Union{Missing,Array{Int64}} = vec_to_mat(confusion_matrix)
        discrimination_calibration_curve::Union{Missing,
                                                Tuple{Vector{Float64},
                                                      Vector{Float64}}}
        discrimination_calibration_score::Union{Missing,Float64}
        multiclass_IRA_kappas::Union{Missing,Float64}
        multiclass_kappa::Union{Missing,Float64}
        optimal_threshold::Union{Missing,Float64}
        optimal_threshold_class::Union{Missing,Int64}
        per_class_IRA_kappas::Union{Missing,Vector{Float64}}
        per_class_kappas::Union{Missing,Vector{Float64}}
        stratified_kappas::Union{Missing,
                                 Vector{NamedTuple{(:per_class,
                                                    :multiclass,
                                                    :n),
                                                   Tuple{Vector{Float64},
                                                         Float64,
                                                         Int64}}}}
        per_class_pr_curves::Union{Missing,
                                   Vector{Tuple{Vector{Float64},
                                                Vector{Float64}}}}
        per_class_reliability_calibration_curves::Union{Missing,
                                                        Vector{Tuple{Vector{Float64},
                                                                     Vector{Float64}}}}
        per_class_reliability_calibration_scores::Union{Missing,
                                                        Vector{Float64}}
        per_class_roc_aucs::Union{Missing,Vector{Float64}}
        per_class_roc_curves::Union{Missing,
                                    Vector{Tuple{Vector{Float64},
                                                 Vector{Float64}}}}
        per_expert_discrimination_calibration_curves::Union{Missing,
                                                            Vector{Tuple{Vector{Float64},
                                                                         Vector{Float64}}}}
        per_expert_discrimination_calibration_scores::Union{Missing,
                                                            Vector{Float64}}
        spearman_correlation::Union{Missing,
                                    NamedTuple{(:ρ, :n,
                                                :ci_lower,
                                                :ci_upper),
                                               Tuple{Float64,
                                                     Int64,
                                                     Float64,
                                                     Float64}}}
        thresholds::Union{Missing,Vector{Float64}}
    end
`Dict` of metrics results (e.g. from Lighthouse <v0.14.0) into an [`EvaluationV1`](@ref).

#TODO change the name of the funtions. We probably wont have *_row anymore.
A Legolas-generated record type representing metrics used in model evaluation,
computed by [`evaluation_metrics_record`](@ref) and [`evaluation_metrics`](@ref).

#TODO what will happen to the constructors?
Constructor that takes `evaluation_row_dict` converts [`evaluation_metrics`](@ref)

See https://github.com/beacon-biosignals/Legolas.jl for details regarding Legolas record types.
"""
EvaluationV1

function EvaluationV1(d::Dict)
    row = (; (Symbol(k) => v for (k, v) in pairs(d))...)
    return EvaluationV1(row)
end

"""
    _evaluation_row_dict(row::EvaluationV1) -> Dict{String,Any}

Convert [`EvaluationV1`](@ref) into `::Dict{String, Any}` results, as are
output by `[`evaluation_metrics`](@ref)` (and predated use of `EvaluationV1` in
Lighthouse <v0.14.0).
"""
function _evaluation_dict(row::EvaluationV1)
    return Dict(string(k) => v for (k, v) in pairs(NamedTuple(row)) if !ismissing(v))
end

#####
##### `Observation
#####

@schema "lighthouse.observation" Observation
@version ObservationV1 begin
    predicted_hard_label::Int64
    predicted_soft_labels::Vector{Float32}
    elected_hard_label::Int64
    votes::Union{Missing,Vector{Int64}}
end

"""
    @version ObservationObjectV1 begin
        predicted_hard_label::Int64
        predicted_soft_labels::Vector{Float32}
        elected_hard_label::Int64
        votes::Union{Missing,Vector{Int64}}
    end
#TODO change the name of the funtions. We probably wont have *_row anymore.
A Legolas-generated record type representing the per-observation input values
required to compute [`evaluation_metrics_record`](@ref).
See https://github.com/beacon-biosignals/Legolas.jl for details regarding Legolas record types.
"""
ObservationV1

# Convert vector of per-class soft label vectors to expected matrix format, e.g.,
# [[0.1, .2, .7], [0.8, .1, .1]] for 2 observations of 3-class classification returns
# ```
# [0.1 0.2 0.7;
# 0.8 0.1 0.1]
# ```
function _predicted_soft_to_matrix(per_observation_soft_labels)
    return transpose(reduce(hcat, per_observation_soft_labels))
end

function _observation_table_to_inputs(observation_table)
    Legolas.validate(Tables.schema(observation_table), ObservationV1SchemaVersion())
    df_table = Tables.columns(observation_table)
    votes = missing
    if any(ismissing, df_table.votes) && !all(ismissing, df_table.votes)
        throw(ArgumentError("`:votes` must either be all `missing` or contain no `missing`"))
    end
    votes = any(ismissing, df_table.votes) ? missing :
            transpose(reduce(hcat, df_table.votes))

    predicted_soft_labels = _predicted_soft_to_matrix(df_table.predicted_soft_labels)
    return (; predicted_hard_labels=df_table.predicted_hard_label, predicted_soft_labels,
            elected_hard_labels=df_table.elected_hard_label, votes)
end

function _inputs_to_observation_table(; predicted_hard_labels::AbstractVector,
                                      predicted_soft_labels::AbstractMatrix,
                                      elected_hard_labels::AbstractVector,
                                      votes::Union{Nothing,Missing,AbstractMatrix}=nothing)
    votes_itr = has_value(votes) ? eachrow(votes) :
                Iterators.repeated(missing, length(predicted_hard_labels))
    predicted_soft_labels_itr = eachrow(predicted_soft_labels)
    if !(length(predicted_hard_labels) ==
         length(predicted_soft_labels_itr) ==
         length(elected_hard_labels) ==
         length(votes_itr))
        throw(DimensionMismatch("Inputs do not all have the same number of observations"))
    end
    observation_table = map(predicted_hard_labels, elected_hard_labels,
                            predicted_soft_labels_itr,
                            votes_itr) do predicted_hard_label, elected_hard_label,
                                          predicted_soft_labels, votes
        return ObservationV1(; predicted_hard_label, elected_hard_label,
                             predicted_soft_labels, votes)
    end
    return observation_table
end

#####
##### Metrics
#####

"""
    Curve(x, y)

Represents a (plot) curve of `x` and `y` points.

When constructing a `Curve`, `missing`'s are replaced with `NaN`, and values are converted to `Float64`.
Curve objects `c` support iteration, `x, y = c`, and indexing, `x = c[1]`, `y = c[2]`.
"""
struct Curve
    x::Vector{Float64}
    y::Vector{Float64}
    function Curve(x::Vector{Float64}, y::Vector{Float64})
        length(x) == length(y) ||
            throw(DimensionMismatch("Arguments to `Curve` must have same length. Got `length(x)=$(length(x))` and `length(y)=$(length(y))`"))
        return new(x, y)
    end
end

floatify(x) = convert(Vector{Float64}, replace(x, missing => NaN))
Curve(x, y) = Curve(floatify(x), floatify(y))
function Curve(t::Tuple)
    length(t) == 2 ||
        throw(ArgumentError("Arguments to `Curve` must consist of x- and y- iterators"))
    return Curve(floatify(first(t)), floatify(last(t)))
end
Curve(c::Curve) = c
Base.iterate(c::Curve, st=1) = st <= fieldcount(Curve) ? (getfield(c, st), st + 1) : nothing
Base.length(::Curve) = fieldcount(Curve)
Base.size(c::Curve) = (fieldcount(Curve), length(c.x))
Base.getindex(c::Curve, i::Int) = getfield(c, i)
for op in (:(==), :isequal)
    @eval function Base.$(op)(c1::Curve, c2::Curve)
        return $op(c1.x, c2.x) && $op(c1.y, c2.y)
    end
end
Base.hash(c::Curve, h::UInt) = hash(:Curve, hash(c.x, hash(c.y, h)))

const CURVE_ARROW_NAME = Symbol("JuliaLang.Lighthouse.Curve")
ArrowTypes.arrowname(::Type{<:Curve}) = CURVE_ARROW_NAME
ArrowTypes.JuliaType(::Val{CURVE_ARROW_NAME}) = Curve

@schema "lighthouse.class" Class
@version ClassV1 begin
    class_index::Union{Int64,Symbol} = check_valid_class(class_index)
    class_labels::Union{Missing,Vector{String}}
end

"""
    @version ClassObjectV1 begin
        class_index::Union{Int64,Symbol} = check_valid_class(class_index)
        class_labels::Union{Missing,Vector{String}} = coalesce(class_labels, missing)
    end

A Legolas-generated record type representing a single column `class_index`
that holds either an integer or the value `:multiclass`, and the class names
associated to the integer class indices.
See https://github.com/beacon-biosignals/Legolas.jl for details regarding Legolas record types.
"""
ClassV1

check_valid_class(class_index::Integer) = Int64(class_index)

function check_valid_class(class_index::Any)
    return class_index === :multiclass ? class_index :
           throw(ArgumentError("Classes must be integer or the symbol `:multiclass`"))
end

@schema "lighthouse.label-metrics" LabelMetrics
@version LabelMetricsV1 > ClassV1 begin
    ira_kappa::Union{Missing,Float64}
    per_expert_discrimination_calibration_curves::Union{Missing,Vector{Curve}} = lift(v -> Curve.(v),
                                                                                      per_expert_discrimination_calibration_curves)
    per_expert_discrimination_calibration_scores::Union{Missing,Vector{Float64}}
end

"""
    @version LabelMetricsObjectV1 begin
        ira_kappa::Union{Missing,Float64}
        per_expert_discrimination_calibration_curves::Union{Missing,Vector{Curve}} = ismissing(per_expert_discrimination_calibration_curves) ?
                                                                                     missing :
                                                                                     Curve.(per_expert_discrimination_calibration_curves)
        per_expert_discrimination_calibration_scores::Union{Missing,Vector{Float64}}
    end

A Legolas-generated record type representing metrics calculated over labels
provided by multiple labelers. See also [`get_label_metrics_multirater`](@ref)
and [`get_label_metrics_multirater_multiclass`](@ref).
See https://github.com/beacon-biosignals/Legolas.jl for details regarding Legolas record types.
"""
LabelMetricsV1

@schema "lighthouse.hardened-metrics" HardenedMetrics
@version HardenedMetricsV1 > ClassV1 begin
    confusion_matrix::Union{Missing,Array{Int64}} = vec_to_mat(confusion_matrix)
    discrimination_calibration_curve::Union{Missing,Curve} = lift(Curve,
                                                                  discrimination_calibration_curve)
    discrimination_calibration_score::Union{Missing,Float64}
    ea_kappa::Union{Missing,Float64}
end

"""
    @version HardenedMetricsObjectV1 > ClassObjectV1 begin
        confusion_matrix::Union{Missing,Array{Int64}} = vec_to_mat(confusion_matrix)
        discrimination_calibration_curve::Union{Missing,Curve} = ismissing(discrimination_calibration_curve) ?
                                                                 missing :
                                                                 Curve(discrimination_calibration_curve)
        discrimination_calibration_score::Union{Missing,
                                                Float64}
        ea_kappa::Union{Missing,Float64}
    end

A Legolas-generated record type representing metrics calculated over predicted
hard labels. See also [`get_hardened_metrics`](@ref),
[`get_hardened_metrics_multirater`](@ref), and [`get_hardened_metrics_multiclass`](@ref).

See https://github.com/beacon-biosignals/Legolas.jl for details regarding Legolas record types.
"""
HardenedMetricsV1

@schema "lighthouse.tradeoff-metrics" TradeOffMetrics
@version TradeOffMetricsV1 > ClassV1 begin
    roc_curve::Curve = lift(Curve, roc_curve)
    roc_auc::Float64
    pr_curve::Curve = lift(Curve, pr_curve)
    spearman_correlation::Union{Missing,Float64}
    spearman_correlation_ci_upper::Union{Missing,Float64}
    spearman_correlation_ci_lower::Union{Missing,Float64}
    n_samples::Union{Missing,Int}
    reliability_calibration_curve::Union{Missing,Curve} = lift(Curve,
                                                               reliability_calibration_curve)
    reliability_calibration_score::Union{Missing,Float64}
end
"""
@version TradeOffMetricsObjectV1 > ClassObjectV1 begin
    roc_curve::Curve = ismissing(roc_curve) ? missing :
    Curve(roc_curve),
    roc_auc::Float64,
    pr_curve::Curve = ismissing(pr_curve) ? missing :
    Curve(pr_curve),
    spearman_correlation::Union{Missing,Float64},
    spearman_correlation_ci_upper::Union{Missing,
        Float64},
        spearman_correlation_ci_lower::Union{Missing,
        Float64},
        n_samples::Union{Missing,Int},
        reliability_calibration_curve::Union{Missing,Curve} = ismissing(reliability_calibration_curve) ?
                                                              missing :
                                                              Curve(reliability_calibration_curve),
                                                              reliability_calibration_score::Union{Missing,
                                                                                                   Float64}

A Legolas-generated record type representing metrics calculated over predicted
metrics calculated over predicted soft labels. See also
[`get_tradeoff_metrics`](@ref) and [`get_tradeoff_metrics_binary_multirater`](@ref).

See https://github.com/beacon-biosignals/Legolas.jl for details regarding Legolas record types.
"""
TradeOffMetricsV1

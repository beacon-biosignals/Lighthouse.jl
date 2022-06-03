@testset "`vec_to_mat`" begin
    mat = [3 5 6; 6 7 8; 9 10 11]
    @test Lighthouse.vec_to_mat(vec(mat)) == mat
    @test Lighthouse.vec_to_mat(mat) == mat
    @test ismissing(Lighthouse.vec_to_mat(missing))
    @test_throws DimensionMismatch Lighthouse.vec_to_mat(collect(1:6)) # Invalid dimensions
end

@testset "`EvaluationRow` basics" begin
    # Most EvaluationRow testing happens via the `test_evaluation_metrics_roundtrip`
    # in test/learn.jl

    # Roundtrip from dict
    dict = Dict("class_labels" => ["foo", "bar"], "multiclass_kappa" => 3)
    test_evaluation_metrics_roundtrip(dict)

    # Handle fun case
    mat_dict = Dict("confusion_matrix" => [3 5 6; 6 7 8; 9 10 11])
    test_evaluation_metrics_roundtrip(mat_dict)
end

function test_roundtrip_observation_table(; kwargs...)
    table = Lighthouse._inputs_to_observation_table(; kwargs...)
    rt_inputs = Lighthouse._observation_table_to_inputs(table)
    @test issetequal(keys(kwargs), keys(rt_inputs))
    for k in keys(kwargs)
        @test isequal(kwargs[k], rt_inputs[k]) || k
    end
    return table
end

@testset "`ObservationRow`" begin
    # Multiclass
    num_observations = 100
    classes = ["A", "B", "C", "D"]
    predicted_soft_labels = rand(StableRNG(22), Float32, num_observations, length(classes))
    predicted_hard_labels = map(argmax, eachrow(predicted_soft_labels))

    # Single labeler: round-trip `ObservationRow``...
    elected_hard_one_labeller = predicted_hard_labels[[1:50..., 1:50...]]  # Force 50% TP overall
    votes = missing
    table = test_roundtrip_observation_table(; predicted_soft_labels, predicted_hard_labels,
                                             elected_hard_labels=elected_hard_one_labeller,
                                             votes)

    # ...and parity in evaluation_metrics calculation:
    metrics_from_inputs = Lighthouse.evaluation_metrics_row(predicted_hard_labels,
                                                            predicted_soft_labels,
                                                            elected_hard_one_labeller,
                                                            classes; votes)
    metrics_from_table = Lighthouse.evaluation_metrics_row(table, classes)
    @test isequal(metrics_from_inputs, metrics_from_table)

    # Multiple labelers: round-trip `ObservationRow``...
    num_voters = 5
    possible_vote_labels = collect(0:length(classes)) # vote 0 == "no vote"
    vote_rng = StableRNG(22)
    votes = [rand(vote_rng, possible_vote_labels)
             for sample in 1:num_observations, voter in 1:num_voters]
    votes[:, 3] .= votes[:, 4] # Voter 4 voted identically to voter 3 (force non-zero agreement)
    elected_hard_multilabeller = map(row -> majority(vote_rng, row, 1:length(classes)),
                                     eachrow(votes))
    table = test_roundtrip_observation_table(; predicted_soft_labels, predicted_hard_labels,
                                             elected_hard_labels=elected_hard_multilabeller,
                                             votes)

    # ...is there parity in evaluation_metrics calculations?
    metrics_from_inputs = Lighthouse.evaluation_metrics_row(predicted_hard_labels,
                                                            predicted_soft_labels,
                                                            elected_hard_multilabeller,
                                                            classes; votes)
    metrics_from_table = Lighthouse.evaluation_metrics_row(table, classes)
    @test isequal(metrics_from_inputs, metrics_from_table)

    r_table = Lighthouse._inputs_to_observation_table(; predicted_soft_labels,
                                                      predicted_hard_labels,
                                                      elected_hard_labels=elected_hard_multilabeller,
                                                      votes)
    @test isnothing(Legolas.validate(r_table, Lighthouse.OBSERVATION_ROW_SCHEMA))

    # ...can we handle both dataframe input and more generic row iterators?
    df_table = DataFrame(r_table)
    output_r = Lighthouse._observation_table_to_inputs(r_table)
    output_df = Lighthouse._observation_table_to_inputs(df_table)
    @test isequal(output_r, output_df)

    # Safety last!
    transform!(df_table, :votes => ByRow(v -> isodd(sum(v)) ? missing : v);
               renamecols=false)
    @test_throws ArgumentError Lighthouse._observation_table_to_inputs(df_table)

    transform!(df_table, :votes => ByRow(v -> ismissing(v) ? [1, 2, 3] : v);
               renamecols=false)
    @test_throws ArgumentError Lighthouse._observation_table_to_inputs(df_table)

    @test_throws DimensionMismatch Lighthouse._inputs_to_observation_table(;
                                                                           predicted_soft_labels,
                                                                           predicted_hard_labels=predicted_hard_labels[1:4],
                                                                           elected_hard_labels=elected_hard_multilabeller,
                                                                           votes)
end

@testset "`ClassRow" begin
    @test isa(Lighthouse.ClassRow(; class_index=3, class_labels=missing).class_index, Int64)
    @test isa(Lighthouse.ClassRow(; class_index=Int8(3), class_labels=missing).class_index, Int64)
    @test Lighthouse.ClassRow(; class_index=:multiclass).class_index == :multiclass
    @test Lighthouse.ClassRow(; class_index=:multiclass, class_labels=["a", "b"]).class_labels == ["a", "b"]

    @test_throws ArgumentError Lighthouse.ClassRow(; class_index=3.0f0, class_labels=missing)
    @test_throws ArgumentError Lighthouse.ClassRow(; class_index=:mUlTiClAsS, class_labels=missing)
end

@testset "class_labels" begin
    predicted_soft_labels = [0.51 0.49
                             0.49 0.51
                             0.1 0.9
                             0.9 0.1
                             0.0 1.0]
    elected_hard_labels = [1, 2, 2, 2, 1]
    predicted_hard_labels = [1,2,2,1,2]
    thresholds = [0.25, 0.5, 0.75]
    i_class = 2
    class_labels = ["a", "b"]
    default_metrics = get_tradeoff_metrics(predicted_soft_labels,
                                           elected_hard_labels,
                                           i_class; thresholds, class_labels)
    @test default_metrics.class_labels == class_labels
end


@testset "`Curve`" begin
    c = ([1, 2, 3], [4, 5, 6])
    @test Curve(c...) isa Curve
    @test Curve(c) isa Curve
    @test Curve(1:8, 2:9) isa Curve
    @test Curve([], []) isa Curve
    @test Curve(Float64[], Float64[]) isa Curve

    @test_throws DimensionMismatch Curve([2], [])
    @test_throws ArgumentError Curve((3, 2, 1))

    @test isequal(Lighthouse.floatify([3, missing, 2.4]), Float64[3, NaN, 2.4])

    curve = Curve(c...)
    @test length(curve) == 2
    @test size(curve) == (2, 3)
    @test isequal(curve, Curve(c...))

    @test first(Arrow.Table(Arrow.tobuffer([curve])).x) == curve.x
    @test first(Arrow.Table(Arrow.tobuffer([curve])).y) == curve.y
end

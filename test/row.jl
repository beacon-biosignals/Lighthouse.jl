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
    table = Lighthouse._inputs_to_obervation_table(; kwargs...)
    rt_inputs = Lighthouse._obervation_table_to_inputs(table)
    @test issetequal(keys(kwargs), keys(rt_inputs))
    for k in keys(kwargs)
        @test isequal(kwargs[k], rt_inputs[k]) || k
    end
    return table, rt_inputs
end

@testset "`ObservationRow`" begin
    # Multiclass
    num_observations = 100
    classes = ["A", "B", "C", "D"]
    predicted_soft_labels = rand(StableRNG(22), Float32, num_observations, length(classes))
    predicted_hard_labels = map(argmax, eachrow(predicted_soft))

    # ...single labeler
    elected_hard_one_labeller = predicted_hard[[1:50..., 1:50...]]  # Force 50% TP overall
    votes = missing #todo: should maybe be "nothing" for serialization roundtrip, but this is what our previous default was...

    table, rt_inputs = test_roundtrip_observation_table(; predicted_soft_labels,
                                                        predicted_hard_labels,
                                                        elected_hard_labels=elected_hard_one_labeller,
                                                        votes)
    # @test isequal(evaluation_metrics_row(table), evaluation_metrics_row(rt_inputs))

    # ...multilabeler
    num_voters = 5
    possible_vote_labels = collect(0:length(classes)) # vote 0 == "no vote"
    vote_rng = StableRNG(22)
    votes = [rand(vote_rng, possible_vote_labels)
             for sample in 1:num_observations, voter in 1:num_voters]
    votes[:, 3] .= votes[:, 4] # Voter 4 voted identically to voter 3 (force non-zero agreement)
    elected_hard_multilabeller = map(row -> majority(vote_rng, row, 1:length(classes)),
                                     eachrow(votes))
    test_roundtrip_observation_table(; predicted_soft_labels, predicted_hard_labels,
                                     elected_hard_labels=elected_hard_multilabeller, votes)

    df_table = Lighthouse._inputs_to_obervation_table(; predicted_soft_labels,
                                                      predicted_hard_labels,
                                                      elected_hard_labels=elected_hard_multilabeller,
                                                      votes)
    @test isa(df_table, DataFrame)
    r_table = [ObservationRow(r) for r in eachrow(df_table)]

    # Can handle both dataframe input and more generic row iterators
    output_r = Lighthouse._obervation_table_to_inputs(r_table)
    output_dft = Lighthouse._obervation_table_to_inputs(df_table)
    @test isequal(output_r, output_dft)

    # Safety last!
    transform!(df_table, :votes => ByRow(v -> isodd(sum(v)) ? missing : v);
               renamecols=false)
    @test_throws ArgumentError Lighthouse._obervation_table_to_inputs(df_table)

    transform!(df_table, :votes => ByRow(v -> ismissing(v) ? [1, 2, 3] : v);
               renamecols=false)
    @test_throws ArgumentError Lighthouse._obervation_table_to_inputs(df_table)

    @test_throws DimensionMismatch Lighthouse._inputs_to_obervation_table(;
                                                                          predicted_soft_labels,
                                                                          predicted_hard_labels=predicted_hard_labels[1:4],
                                                                          elected_hard_labels=elected_hard_multilabeller,
                                                                          votes)
end

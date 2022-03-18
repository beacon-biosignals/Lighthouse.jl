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
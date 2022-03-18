@testset "`vec_to_mat`" begin
    mat = [3 5 6; 6 7 8; 9 10 11]
    @test Lighthouse.vec_to_mat(vec(mat)) == mat
    @test Lighthouse.vec_to_mat(mat) == mat
    @test ismissing(Lighthouse.vec_to_mat(missing))
    @test_throws DimensionMismatch Lighthouse.vec_to_mat(collect(1:6)) # Invalid dimensions
end

@testset "`EvaluationRow`" begin
    # Basic roundtrip
    dict = Dict("class_labels" => ["foo", "bar"], "multiclass_kappa" => 3)
    test_evaluation_metrics_roundtrip(dict)

    # Don't lose extra columns (basic Legolas functionality)
    extra_dict = Dict("class_labels" => ["foo", "bar"], "multiclass_kappa" => 3,
                      "rabbit" => 2432)
    test_evaluation_metrics_roundtrip(extra_dict)

    # Handle fun cases
    mat_dict = Dict("confusion_matrix" => [3 5 6; 6 7 8; 9 10 11])
    test_evaluation_metrics_roundtrip(mat_dict)
end
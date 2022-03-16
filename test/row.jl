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
    @test test_roundtrip_evaluation(dict)

    # Should ignore any additional fields that we don't convert
    extra_dict = Dict("class_labels" => ["foo", "bar"], "multiclass_kappa" => 3,
                      "rabbit" => 2432)
    @test test_roundtrip_evaluation(extra_dict)

    mat_dict = Dict("confusion_matrix" => [3 5 6; 6 7 8; 9 10 11])
    mat_row = Lighthouse.EvaluationRow(mat_dict)
    rt_row = roundtrip_row(mat_row)
end
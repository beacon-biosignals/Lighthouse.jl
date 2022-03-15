@testset `vec_to_mat` begin
    mat = [3 5 6; 6 7 8; 9 10 11]
    @test Lighthouse.vec_to_mat(vec(mat)) == mat
    @test Lighthouse.vec_to_mat(mat) == mat
    @test ismissing(Lighthouse.vec_to_mat(missing))
    @test_throws DimensionMismatch Lighthouse.vec_to_mat(collect(1:6)) # Invalid dimensions
end

@testset `round trip EvaluationRow tests` begin
    # Basic roundtrip
    dict = Dict("class_labels" => ["foo", "bar"], "multiclass_kappa" => 3)
    row = Lighthouse.evaluation_row(dict)
    @test isa(row, Lighthouse.EvaluationRow)
    @test isequal(Lighthouse._evaluation_row_dict(row), dict)

    # Should ignore any additional fields that we don't convert
    extra_dict = Dict("class_labels" => ["foo", "bar"], "multiclass_kappa" => 3, "rabbit" => 2432)
    row = Lighthouse.evaluation_row(dict)
    @test isa(row, Lighthouse.EvaluationRow)
    @test isequal(Lighthouse._evaluation_row_dict(row), dict)


end
@testset "extract_common_entries" begin
    common_keys = ("class_labels",)
    consistent_results = [
        (; name="model1", data=Dict("class_labels" => ["hi", "bye"], "unrelated"=>1)),
        (; name="model2", data=Dict("class_labels" => ["hi", "bye"], "unrelated"=>2))
    ]
    @test extract_common_entries(consistent_results, common_keys) == Dict("class_labels" => ["hi", "bye"])

    inconsistent_results = [
        (; name="model1", data=Dict("class_labels" => ["hi", "bye"], "unrelated"=>1)),
        (; name="model2", data=Dict("class_labels" => ["hi", "goodbye"], "unrelated"=>2))
    ]
    @test_throws ArgumentError extract_common_entries(inconsistent_results, common_keys)
end

@testset "binary_comparison_metrics_plot" begin
    m1 = (; name = "Model1", data = Dict("class_labels" => ["a", "b"],
                                         "optimal_threshold_class" => 1,
                                         ""))

end
